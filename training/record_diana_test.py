import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path

samplerate = 16000
duration = 2

OUTPUT_DIR = Path("test/positive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("       DIANA v0.1 UNSEEN POSITIVE TEST RECORDING")
print("=" * 60)
print()
print("Record 30 NEW examples of the word 'Diana'.")
print("Do NOT copy or reuse your training recordings.")
print()

for i in range(1, 31):
    input(f"Press Enter and say 'Diana' ({i}/30)...")

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    filename = OUTPUT_DIR / f"diana_test_{i:03}.wav"

    write(
        filename,
        samplerate,
        recording
    )

    print(f"Saved: {filename}")
    print()

print("Positive test recording complete.")