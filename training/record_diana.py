import sounddevice as sd
from scipy.io.wavfile import write

samplerate = 16000
duration = 2

for i in range(1, 51):
    input(f"Press Enter and say 'Diana' ({i}/50)...")

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    filename = f"positive/diana_{i:03}.wav"
    write(filename, samplerate, recording)

    print(f"Saved: {filename}")
