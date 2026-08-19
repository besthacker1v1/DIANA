import os
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 2
OUTPUT_DIR = "negative"

phrases = [
    "Deanna",
    "Dina",
    "Anna",
    "Diana here",
    "Indiana",
    "Indian",
    "banana",
    "begin",
    "computer",
    "hello",
    "good morning",
    "open Chrome",
    "open VS Code",
    "open Spotify",
    "start coding",
    "close the window",
    "play a movie",
    "what time is it",
    "what are you doing",
    "how are you",
    "I am ready",
    "let's go",
    "open my project",
    "show me the files",
    "search the folder",
    "turn it up",
    "turn it down",
    "mute the computer",
    "play music",
    "stop the music",
    "check the system",
    "open the browser",
    "launch the terminal",
    "run the server",
    "fix the code",
    "where is my project",
    "find the file",
    "go back",
    "go home",
    "good night",
    "thank you",
    "okay computer",
    "hey computer",
    "listen",
    "come here",
    "what happened",
    "that is interesting",
    "try again",
    "never mind",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 45)
print("       DIANA NEGATIVE DATASET")
print("=" * 45)
print()
print("You will record 50 NON-DIANA phrases.")
print("Speak naturally at roughly normal volume.")
print()

input("Press ENTER to begin...")

for i, phrase in enumerate(phrases, start=1):
    input(f"\n[{i}/{len(phrases)}] Press ENTER, then say: \"{phrase}\"")

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    filename = os.path.join(OUTPUT_DIR, f"negative_{i:03d}.wav")
    write(filename, SAMPLE_RATE, audio)

    print(f"Saved: {filename}")

print("\nNegative dataset recording complete.")