import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer


class SpeechToText:
    SAMPLE_RATE = 16000

    def __init__(self, model_path):
        print("Loading speech recognition model...")

        self.model = Model(model_path)

        print("Speech recognition model loaded.")

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"[Audio] {status}")

        self.audio_queue.put(bytes(indata))

    def listen(self):
        print("\nDIANA is listening...")

        self.audio_queue = queue.Queue()

        # New recognizer for every command
        recognizer = KaldiRecognizer(
            self.model,
            self.SAMPLE_RATE
        )

        with sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=4000,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            while True:
                data = self.audio_queue.get()

                if recognizer.AcceptWaveform(data):
                    result = json.loads(
                        recognizer.Result()
                    )

                    text = result.get("text", "").strip()

                    if text:
                        return text