import sounddevice as sd

DEVICE = 1
DURATION = 5
SAMPLE_RATE = 16000

print("DIANA microphone test")
print("---------------------")
print("Speak normally for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=DEVICE,
)

sd.wait()

print("Recording complete.")
print(f"Captured {len(audio)} audio samples.")
print("Microphone is working.")