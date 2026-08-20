import subprocess


class CommandExecutor:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def execute(self, command):
        command = command.lower().strip()

        intent = self._detect_intent(command)

        if intent == "open_edge":
            return self._open_edge()

        if intent == "open_chrome":
            return self._open_chrome()

        print(f"DIANA: I don't know how to do: {command}")
        return False

    def _detect_intent(self, command):

        edge_phrases = [
            "open edge",
            "open microsoft edge",
            "launch edge",
            "launch microsoft edge",
            "start edge",
            "start microsoft edge",
        ]

        chrome_phrases = [
            "open chrome",
            "launch chrome",
            "start chrome",
            "open google chrome",
            "launch google chrome",
        ]

        if any(phrase in command for phrase in edge_phrases):
            return "open_edge"

        if any(phrase in command for phrase in chrome_phrases):
            return "open_chrome"

        return None

    def _open_edge(self):
        print("DIANA: Opening Microsoft Edge...")

        if self.dry_run:
            print("[DRY RUN] Would launch: msedge")
            return True

        subprocess.Popen(
            ["cmd", "/c", "start", "", "msedge"],
            shell=False,
        )

        return True

    def _open_chrome(self):
        print("DIANA: Opening Chrome...")

        if self.dry_run:
            print("[DRY RUN] Would launch: chrome")
            return True

        subprocess.Popen(
            ["cmd", "/c", "start", "", "chrome"],
            shell=False,
        )

        return True