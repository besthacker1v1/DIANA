import os
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 2
TOTAL_RECORDINGS = 50

OUTPUT_DIR = "positive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 40)
print("        DIANA VOICE DATASET")
print("=" * 40)
print()
print("You will record 50 wake-word samples.")
print()
print("Say:")
print("    Diana")
print()
print("Speak naturally. Don't force the pronunciation.")
print("Vary your volume and distance slightly.")
print()
input("Press ENTER when you're ready...")

for i in range(1, TOTAL_RECORDINGS + 1):

    input(f"\n[{i}/{TOTAL_RECORDINGS}] Press ENTER, then say 'Diana'...")

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    filename = os.path.join(
        OUTPUT_DIR,
        f"diana_{i:03d}.wav"
    )

    write(filename, SAMPLE_RATE, audio)

    print(f"Saved: {filename}")

print()
print("=" * 40)
print("Dataset recording complete.")
print("=" * 40)