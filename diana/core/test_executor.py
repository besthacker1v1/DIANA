from command_executor import CommandExecutor


def main():
    executor = CommandExecutor(dry_run=True)

    test_commands = [
        # Edge
        "open edge",
        "open microsoft edge",
        "launch edge",
        "start edge",

        # Chrome
        "open chrome",
        "launch chrome",
        "start chrome",
        "open google chrome",

        # Notepad
        "open notepad",
        "open notes pad",

        # Calculator
        "open calculator",
        "open calc",

        # VS Code
        "open vs code",
        "open visual studio code",

        # Unknown
        "open firefox",
    ]

    print("=" * 60)
    print("DIANA COMMAND EXECUTOR TEST")
    print("=" * 60)

    for command in test_commands:
        print(f"\nTEST: {command}")

        result = executor.execute(command)

        print(f"RESULT: {result}")


if __name__ == "__main__":
    main()