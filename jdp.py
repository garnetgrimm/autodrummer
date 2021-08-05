import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.framework.ops import disable_eager_execution
import os

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

model = make_model(input_shape=(129, 1, 3), num_classes=len(data_labels))
keras.utils.plot_model(model, show_shapes=True)
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

#model.summary()

if(os.path.isfile(save_name)):
    model.load_weights(save_name)

if(True):
    checkpoint = ModelCheckpoint(save_name, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    sampleDir = "output"
    specDir =  os.path.join(sampleDir, "specs")
    dataset = image_dataset_from_directory(specDir, labels='inferred', image_size=(129,1))
    history = model.fit(dataset, epochs=10, callbacks=[checkpoint])

img = keras.preprocessing.image.load_img(
    "51_test.jpg", target_size=(129, 1, 3)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
for i in range(len(predictions[0])):
    print(str(data_labels[i]) + ": " + str(round(100*predictions[0][i], 2)) + "%")