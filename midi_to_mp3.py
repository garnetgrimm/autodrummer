from midi2audio import FluidSynth
import glob
import subprocess
import os
from pathlib import Path

fs = FluidSynth('sf2.sf2')

inDir = "samples"
outDir = "output"

def midi_to_mp3(f):
    filename = os.path.basename(f)
    outPath = outDir + f[len(inDir):-len(filename)]
    Path(outPath).mkdir(parents=True, exist_ok=True)
    wavFile = outPath + " " + filename.replace(".mid",".wav")
    mp3File = outPath + " " + filename.replace(".mid",".mp3")
    fs.midi_to_audio(f, wavFile)
    subprocess.run(["ffmpeg", "-y", "-i", wavFile, "-acodec", "libmp3lame", mp3File])
    os.remove(wavFile)

for f in glob.glob(inDir + '/**/*.mid', recursive=True):
    midi_to_mp3(f)
#midi_to_mp3("test.mid")