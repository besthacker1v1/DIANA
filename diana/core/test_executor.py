from command_executor import CommandExecutor


def main():
    executor = CommandExecutor(dry_run=True)

    test_commands = [
        "open edge",
        "open microsoft edge",
        "launch edge",
        "launch microsoft edge",
        "start edge",
        "start microsoft edge",
        "open chrome",
        "launch chrome",
        "start chrome",
        "open google chrome",
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