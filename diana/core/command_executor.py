import subprocess


class CommandExecutor:

    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def _launch(self, command):
        if self.dry_run:
            print(f"[DRY RUN] Would launch: {command}")
            return

        subprocess.Popen(
            command,
            shell=False,
        )

    def execute(self, command):
        command = command.lower().strip()

        # -------------------------
        # Microsoft Edge
        # -------------------------
        if any(phrase in command for phrase in [
            "open edge",
            "open microsoft edge",
            "launch edge",
            "launch microsoft edge",
            "start edge",
            "start microsoft edge",
        ]):
            print("DIANA: Opening Microsoft Edge...")

            self._launch(
                ["cmd", "/c", "start", "", "msedge"]
            )

            return True

        # -------------------------
        # Google Chrome
        # -------------------------
        if any(phrase in command for phrase in [
            "open chrome",
            "open google chrome",
            "launch chrome",
            "launch google chrome",
            "start chrome",
            "start google chrome",
        ]):
            print("DIANA: Opening Chrome...")

            self._launch(
                ["cmd", "/c", "start", "", "chrome"]
            )

            return True

        # -------------------------
        # Notepad
        # -------------------------
        if any(phrase in command for phrase in [
            "open notepad",
            "open notes pad",
            "launch notepad",
            "launch notes pad",
            "start notepad",
            "start notes pad",
        ]):
            print("DIANA: Opening Notepad...")

            self._launch(["notepad.exe"])

            return True

        # -------------------------
        # Calculator
        # -------------------------
        if any(phrase in command for phrase in [
            "open calculator",
            "open calc",
            "launch calculator",
            "launch calc",
            "start calculator",
            "start calc",
        ]):
            print("DIANA: Opening Calculator...")

            self._launch(["calc.exe"])

            return True

        # -------------------------
        # Visual Studio Code
        # -------------------------
        if any(phrase in command for phrase in [
            "open vs code",
            "open visual studio code",
            "launch vs code",
            "launch visual studio code",
            "start vs code",
            "start visual studio code",
        ]):
            print("DIANA: Opening Visual Studio Code...")

            self._launch(["code"])

            return True

        # -------------------------
        # Unknown command
        # -------------------------
        print(f"DIANA: I don't know how to do: {command}")
        return False