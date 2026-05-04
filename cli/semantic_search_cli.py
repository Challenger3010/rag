import argparse
from utils.commands import verify_command, embed_text_command, verify_embeddings_command, embed_query_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Verify embedding model")
    subparser.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embeder = subparser.add_parser("embed_text", help="Embed text")
    query_embeder = subparser.add_parser("embed_query", help="embed query")

    embeder.add_argument("text", type=str, help="Text to be embedded")
    query_embeder.add_argument("text", type=str, help="Text to be embedded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_command()
        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings_command()
        case "embed_query":
            embed_query_command(args.text)

        
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()