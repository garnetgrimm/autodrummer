import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import subprocess
import numpy as np
import os
from mido import MidiFile, MetaMessage, tick2second, tempo2bpm
from PIL import Image as im
from pathlib import Path
from midi2audio import FluidSynth
import glob

sampleDir = "output"
midiDir = "midi"

fs = FluidSynth('sf2.sf2')

def midi_to_mp3(midiFile, mp3File):
    wavFile = mp3File.replace(".mp3", ".wav")
    fs.midi_to_audio(midiFile, wavFile)
    subprocess.run(["ffmpeg", "-y", "-i", wavFile, "-acodec", "libmp3lame", mp3File])
    os.remove(wavFile)

def mp3toSpec(mp3File, sampleRate):
    wavFile = mp3File.replace(".mp3", ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", mp3File, "-acodec", "pcm_s32le", "-ar", str(sampleRate), "-ac", "1", wavFile])
    sample_rate, samples = wavfile.read(wavFile)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    spectrogram = np.log(spectrogram)
    os.remove(wavFile)
    ramp_n = 0.5
    spectrogram = np.power(spectrogram, 1 - ramp_n)
    spectrogram = np.multiply(spectrogram, pow(255.0, ramp_n))    
    return spectrogram

def midiToSpec(midiFile, xLen):
    midi = MidiFile(midiFile)
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
    return spectrogram

def arrayToImage(array, name):
    array = array.astype(np.uint8)
    image = im.fromarray(array)
    image.convert("L")
    image.save(name)

def drawSpec(inDir, inFile):
    spectrogram = mp3toSpec(inDir, inFile, 44100)
    midigram = midiToSpec(inDir, inFile, spectrogram.shape[1])
    arrayToImage(spectrogram, os.path.join(sampleDir, inDir, inFile, "spectrogram.png"))
    arrayToImage(midigram, os.path.join(sampleDir, inDir, inFile, "midigram.png"))

def drawSplitSpec(midiFile, mp3File, specDir, seperate=True, formatString="{note}_image_{fileHash}_{i}_{n}_.jpg"):
    sampleRate = 44100
    spectrogram = mp3toSpec(mp3File, sampleRate)
    midigram = midiToSpec(midiFile, spectrogram.shape[1])    
    midi = MidiFile(midiFile, clip=True)
    xLen = spectrogram.shape[1]
    slices = []
    currTime = 0
    totalTime = 0
    for msg in midi:
        totalTime +=  msg.time
    for msg in midi:
        deltaTime = msg.time
        if not msg.is_meta and msg.velocity != 0:
            startSlice = int((currTime/totalTime) * xLen)
            endSlice = int(((currTime+deltaTime)/totalTime) * xLen)
            if(endSlice - startSlice <= 0): continue
            slices.append((msg.note, spectrogram[:,startSlice:endSlice]))
        currTime += deltaTime    
    fileHash = abs(hash(os.path.basename(mp3File))) % (10 ** 8)
    for i, slice in enumerate(slices):
        note = slice[0]
        slice = slice[1]
        if(seperate):
            fileDir = os.path.join(specDir, str(note))
        else:
            fileDir = specDir
        Path(fileDir).mkdir(parents=True, exist_ok=True)
        for n in range(slice.shape[1]-1):
            basename = formatString.format(fileHash=fileHash, i=i, n=n, note=note)
            filename = os.path.join(fileDir, basename)
            arrayToImage(slice[:,n:n+1], filename)

if __name__ == "__main__":
    for midiFile in glob.glob(midiDir + '/**/*.mid', recursive=True):
        outPath = sampleDir + midiFile[len(midiDir):-4]
        Path(outPath).mkdir(parents=True, exist_ok=True)
        mp3File = os.path.join(outPath, "sample.mp3")
        midi_to_mp3(midiFile, mp3File)
        drawSplitSpec(midiFile, mp3File, os.path.join(sampleDir, "specs"))