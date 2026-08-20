import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path

samplerate = 16000
duration = 2

OUTPUT_DIR = Path("test/negative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_PHRASES = [
    "Dina",
    "Deanna",
    "Indiana",
    "Daniel",
    "David",
    "computer",
    "banana",
    "Diana is here",
    "I need help",
    "open the computer",
    "what time is it",
    "how are you",
    "good morning",
    "come here",
    "open VS Code",
    "random sentence",
    "hello there",
    "what are you doing",
    "this is a test",
    "I am going home",
    "turn on the computer",
    "play some music",
    "where are you",
    "let's continue",
    "can you hear me",
    "open the browser",
    "close the window",
    "I need to study",
    "computer science",
    "good evening",
]

print("=" * 60)
print("       DIANA v0.1 UNSEEN NEGATIVE TEST RECORDING")
print("=" * 60)
print()
print("These recordings must NOT contain the wake word by itself.")
print("Say the displayed phrase naturally.")
print()

for i, phrase in enumerate(NEGATIVE_PHRASES, start=1):

    input(
        f"Press Enter and say: \"{phrase}\" "
        f"({i}/{len(NEGATIVE_PHRASES)})..."
    )

    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    filename = OUTPUT_DIR / f"negative_test_{i:03}.wav"

    write(
        filename,
        samplerate,
        recording
    )

    print(f"Saved: {filename}")
    print()

print("=" * 60)
print("Negative test recording complete.")
print("=" * 60)