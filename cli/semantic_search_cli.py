import argparse
from utils.commands import verify_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Verify embedding model")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()
        
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()