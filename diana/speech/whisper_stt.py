import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel


class WhisperSpeechToText:

    SAMPLE_RATE = 16000
    RECORD_SECONDS = 5

    def __init__(self, model_size="base.en"):
        print("Loading Whisper model...")

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        print("Whisper model loaded.")

    def listen(self):
        print("\nDIANA is listening...")

        audio = sd.rec(
            int(self.RECORD_SECONDS * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )

        sd.wait()

        audio = np.squeeze(audio)

        print("Transcribing...")

        segments, _ = self.model.transcribe(
            audio,
            beam_size=5,
            language="en",
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        if text:
            print(f"DIANA heard: {text}")

        return text