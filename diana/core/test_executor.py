import subprocess


class CommandExecutor:

    def execute(self, command):
        command = command.lower().strip()

        if "open edge" in command or "open microsoft edge" in command:
            print("DIANA: Opening Microsoft Edge...")
            subprocess.Popen(
                ["cmd", "/c", "start", "", "msedge"],
                shell=False,
            )
            return True

        if "open chrome" in command:
            print("DIANA: Opening Chrome...")
            subprocess.Popen(
                ["cmd", "/c", "start", "", "chrome"],
                shell=False,
            )
            return True

        print(f"DIANA: I don't know how to do: {command}")
        return False