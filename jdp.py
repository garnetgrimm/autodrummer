import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.framework.ops import disable_eager_execution
import os
from preprocess import drawSplitSpec, midi_to_mp3
import shutil

save_name = "mymodel.hdf5"
data_labels = os.listdir(os.path.join("output", "specs"))

data_augmentation = keras.Sequential(
    [
    ]
)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def train(model):
    checkpoint = ModelCheckpoint(save_name, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    specDir =  os.path.join("output", "specs")
    dataset = image_dataset_from_directory(specDir, labels='inferred', image_size=(129,1))
    history = model.fit(dataset, epochs=10, callbacks=[checkpoint])

def predict(model, img_array):
    predictions = model.predict(img_array)
    pretty_preditictions = dict()
    for i in range(len(predictions[0])):
        pretty_preditictions[int(data_labels[i])] = predictions[0][i]
    return pretty_preditictions
    
def writeSong(predictionList):
    from mido import Message, MidiFile, MidiTrack

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    onNotes = set()
    last_i = 0
    for i,prediction in enumerate(predictionList):
        for note in prediction:
            if(prediction[note] > 0.9):
                if(note not in onNotes):
                    track.append(Message('note_on', channel=9, note=note, velocity=int(prediction[note]*127), time=last_i))
                    onNotes.add(note)
                    last_i = 0
            else:
                if(note in onNotes):
                    track.append(Message('note_on', channel=9, note=note, velocity=0, time=last_i))
                    onNotes.remove(note)
                    last_i = 0
            last_i += 1

    return mid

def validate(model):
    mp3File = os.path.join("validate", "sample.mp3")
    midiFile = os.path.join("midi", "Blues", "12-Bar Blues", "180 Driving Ride F1 S.mid")
    midi_to_mp3(midiFile, mp3File)
    drawSplitSpec(midiFile, mp3File, os.path.join("validate", "specs"), seperate=False, formatString="spectrogram_{i}.jpg")
    validateSpecsDir=os.path.join("validate", "specs")
    grams = os.listdir(validateSpecsDir)
    predictions = []
    for file in grams:
        img = keras.preprocessing.image.load_img(
            os.path.join(validateSpecsDir, file), target_size=(129, 1, 3)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        predictions.append(predict(model, img_array))
    song = writeSong(predictions)
    song.save(os.path.join("validate", "new_song.mid"))
    shutil.copy(midiFile, os.path.join("validate", "old_song.mid"))

# create model
model = make_model(input_shape=(129, 1, 3), num_classes=len(data_labels))
#keras.utils.plot_model(model, show_shapes=True)

# compile model
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

# load weights
if(os.path.isfile(save_name)):
    model.load_weights(save_name)

# train model
#train(model)

validate(model)

