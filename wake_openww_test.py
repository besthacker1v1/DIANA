import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

print("=" * 60)
print("       DIANA - openWakeWord DIAGNOSTIC")
print("=" * 60)
print()
print("Loading model...")

model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx",
)

print("Model loaded.")
print()
print("Speak into the microphone.")
print("Say 'Hey Jarvis' several times.")
print("Press Ctrl+C to stop.")
print()

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    ) as stream:

        counter = 0

        while True:
            audio, _ = stream.read(CHUNK_SIZE)
            audio = np.squeeze(audio)

            prediction = model.predict(audio)
            score = prediction.get("hey_jarvis", 0.0)

            counter += 1

            if counter % 10 == 0:
                print(f"score: {score:.4f}")

except KeyboardInterrupt:
    print("\nStopped.")