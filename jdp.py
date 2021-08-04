import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import subprocess
import numpy as np
import os

sampleDir = "output\\"
midiDir = "samples\\"

def mp3toSpec(inDir, inFile, sampleRate):
    wavFile = sampleDir + inDir + " " + inFile + ".wav"
    mp3File = sampleDir + inDir + " " + inFile + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-i", mp3File, "-acodec", "pcm_s32le", "-ar", str(sampleRate), "-ac", "1", wavFile])
    sample_rate, samples = wavfile.read(wavFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    spectrogram = np.log(spectrogram)
    os.remove(wavFile)
    return frequencies, times, spectrogram

from mido import MidiFile, MetaMessage, tick2second, tempo2bpm
import time
def midiToSpec(inDir, inFile, xLen):
    midFile = midiDir + inDir + inFile + ".mid"
    midi = MidiFile(midFile, clip=True)
    spectrogram = np.zeros((127, xLen))
    currTime = 0
    totalTime = 0
    for msg in midi:
        totalTime +=  msg.time
    for msg in midi:
        deltaTime = msg.time
        if not msg.is_meta:
            for i in range(int((currTime/totalTime) * xLen), xLen):
                spectrogram[msg.note][i] = (msg.velocity/127)*255
        currTime += deltaTime
    print(type(spectrogram))
    print(spectrogram.shape)
    return spectrogram


from PIL import Image as im
def drawSpec(inDir, inFile):
    frequencies, times, spectrogram = mp3toSpec(inDir, inFile, 44100)
    midigram = midiToSpec(inDir, inFile, times.shape[0])
    midis = np.array([i for i in range(127)])
    
    ramp_n = 0.5
    spectrogram = np.power(spectrogram, 1 - ramp_n)
    spectrogram = np.multiply(spectrogram, pow(255.0, ramp_n))
    spectrogram = spectrogram.astype(np.uint8)
    print(spectrogram)
    midigram = midigram.astype(np.uint8)

    specImage = im.fromarray(spectrogram)
    specImage.convert("L")
    specImage.save('spectrogram.png')
    midiImage = im.fromarray(midigram)
    midiImage.convert("L")
    midiImage.save('midigram.png')
    
drawSpec('Classic Rock\\Song 03 146 Jail Breakin\\Grooves\\', '146 S03 Chorus Live 1')