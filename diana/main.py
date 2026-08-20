from diana.wake_engine import WakeEngine
from diana.speech.whisper_stt import WhisperSpeechToText
from diana.core.command_loop import CommandLoop


WAKE_WORD = "alexa"


def main():

    print("=" * 60)
    print("                 DIANA")
    print("=" * 60)

    print("\nInitializing DIANA...\n")

    wake_engine = WakeEngine(
        wake_word=WAKE_WORD,
        threshold=0.5,
    )

    stt = WhisperSpeechToText("base.en")

    command_loop = CommandLoop(stt)

    print("\nDIANA is online.")
    print("Say 'Alexa' to wake her.")

    while True:

        try:
            score = wake_engine.listen()

            print(f"\n[DIANA WAKE] Detected! Score: {score:.3f}")
            print("DIANA: Yes?")

            command = command_loop.listen_for_command()

            if command:
                print(f"\n[COMMAND] {command}")

        except KeyboardInterrupt:
            print("\n\nDIANA shutting down.")
            break


if __name__ == "__main__":
    main()