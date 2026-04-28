import argparse

from utils.commands import search_command, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser("build", help="Build the inverted index")

    search_parser = subparser.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            res = search_command(args.query)
            for i,res in enumerate(res, 1):
                print(f"{i}. {res['title']}")

        case "build":
            build_command()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()