from stt import SpeechToText


MODEL_PATH = (
    "diana/speech/models/vosk-model-small-en-us-0.15"
)


def main():
    stt = SpeechToText(MODEL_PATH)

    while True:
        try:
            text = stt.listen()

            print(f"YOU: {text}")

        except KeyboardInterrupt:
            print("\nSTT stopped.")
            break


if __name__ == "__main__":
    main()