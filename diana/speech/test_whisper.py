import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def main():
    print("Loading Whisper model...")

    model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
    )

    print("Whisper model loaded.\n")

    print(f"Speak for {RECORD_SECONDS} seconds...")
    print("Try:")
    print("  open Chrome")
    print("  what time is it")
    print("  hello Diana")
    print()

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    audio = np.squeeze(audio)

    print("\nTranscribing...")

    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )

    print(f"\nYOU: {text}")
    print("\nTest complete.")


if __name__ == "__main__":
    main()