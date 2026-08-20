from diana.wake_engine import WakeEngine
from diana.speech.whisper_stt import WhisperSpeechToText
from diana.core.command_loop import CommandLoop
from diana.core.command_executor import CommandExecutor

# Configuration Constants
WAKE_WORD = "alexa"
WAKE_THRESHOLD = 0.5
WHISPER_MODEL = "base.en"


def initialize_components():
    """Instantiate and return the required DIANA sub-modules."""
    wake_engine = WakeEngine(wake_word=WAKE_WORD, threshold=WAKE_THRESHOLD)
    stt = WhisperSpeechToText(model_size=WHISPER_MODEL)
    command_loop = CommandLoop(stt)
    executor = CommandExecutor()
    
    return wake_engine, command_loop, executor


def run_pipeline(wake_engine, command_loop, executor):
    """Main execution loop for listening and processing commands."""
    print("Wait for wake word... Say 'Alexa' to activate.")

    while True:
        try:
            score = wake_engine.listen()
            print(f"\n[DIANA WAKE] Detected! Score: {score:.3f}")
            print("DIANA: Yes?")

            command = command_loop.listen_for_command()

            if command:
                print(f"\n[COMMAND] {command}")
                executor.execute(command)

        except KeyboardInterrupt:
            print("\n\nDIANA shutting down cleanly.")
            break
        except Exception as err:
            print(f"\n[ERROR] An unexpected error occurred: {err}")


def main():
    print("=" * 60)
    print(" " * 24 + "DIANA")
    print("=" * 60)
    print("\nInitializing DIANA...\n")

    wake_engine, command_loop, executor = initialize_components()

    print("\nDIANA is online.")
    run_pipeline(wake_engine, command_loop, executor)


if __name__ == "__main__":
    main()