import argparse
from utils.commands import verify_command, embed_text_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Verify embedding model")

    embeder = subparser.add_parser("embed_text", help="Verify embedding model")
    embeder.add_argument("text", type=str, help="Text to be embedded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()
        case "embed_text":
            embed_text_command(args.text)
        
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()