import numpy as np
import sounddevice as sd

from openwakeword.model import Model


class WakeEngine:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1280

    def __init__(
        self,
        wake_word="hey_jarvis",
        threshold=0.5,
    ):
        self.wake_word = wake_word
        self.threshold = threshold

        print("Loading wake-word model...")

        self.model = Model(
            wakeword_models=[wake_word],
            inference_framework="onnx",
        )

        print("Wake-word model loaded.")

    def listen(self):
        print("Wake engine listening...")

        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.CHUNK_SIZE,
        ) as stream:

            while True:
                audio, _ = stream.read(self.CHUNK_SIZE)

                audio = np.squeeze(audio)

                predictions = self.model.predict(audio)

                score = predictions.get(
                    self.wake_word,
                    0.0,
                )

                if score >= self.threshold:
                    return score