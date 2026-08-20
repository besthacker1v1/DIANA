class CommandLoop:

    def __init__(self, stt):
        self.stt = stt

    def listen_for_command(self):
        print("\nDIANA: Listening for your command...")

        text = self.stt.listen()

        if text:
            print(f"DIANA heard: {text}")
            return text

        print("DIANA: I didn't catch that.")
        return None