import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


DEVICE = 1
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1280

WAKE_THRESHOLD = 0.8
COOLDOWN_SECONDS = 2.0


print("================================")
print("        DIANA WAKE ENGINE       ")
print("================================")
print("Loading neural wake detector...")

model = Model(
    wakeword_models=["hey_jarvis"]
)

print("Wake detector online.")
print("Listening for wake word...")
print("Test wake word: Hey Jarvis")
print()


last_wake_time = 0


def audio_callback(indata, frames, time_info, status):
    global last_wake_time

    if status:
        print("Audio status:", status)

    audio = np.squeeze(indata.copy())

    predictions = model.predict(audio)

    for name, score in predictions.items():

        current_time = time.time()

        if (
            score >= WAKE_THRESHOLD
            and current_time - last_wake_time >= COOLDOWN_SECONDS
        ):
            last_wake_time = current_time

            print()
            print("================================")
            print("       🔔 DIANA ACTIVATED")
            print("================================")
            print(f"Wake confidence: {score:.3f}")
            print()


try:

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=DEVICE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
    ):

        while True:
            time.sleep(0.1)

except KeyboardInterrupt:

    print()
    print("DIANA shutting down.")