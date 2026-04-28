import argparse
from InvertedIndex import InvertedIndex


from utils.keyword_search_utils import search_command

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
            print(f"Building the inverted index...")
            indexer = InvertedIndex()
            indexer.build()
            print(f"Saving the index...")
            indexer.save()
            docs = indexer.get_document("merida")
            print("index saved")



            print(f"First document for token 'merida' = {docs[0]}")
           
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()