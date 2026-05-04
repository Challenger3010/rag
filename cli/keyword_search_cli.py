import argparse

from utils.search_utils import BM25_K1, BM25_B
from utils.commands import search_command, build_command, tf_command, idf_command, tfidf_command, bm25_idf_command, bm25_tf_command, bm25search_command, verify_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("build", help="Build the inverted index")

    search_parser = subparser.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    frequency_parser = subparser.add_parser("tf", help="Look for frequency of a term in a document")
    frequency_parser.add_argument("doc_id", type=int, help="Document ID")
    frequency_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparser.add_parser("idf", help="Get IDF for a term")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparser.add_parser("tfidf", help="TBD")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    # create bm25 parsers with args
    bm25idf_parser = subparser.add_parser("bm25idf", help="Get IDF for a term")
    bm25idf_parser.add_argument("term", type=str, help="Term")

    bm25_tf_parser = subparser.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparser.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:

        case "search":
            print(f"Searching for: {args.query}")
            res = search_command(args.query)
            for i,res in enumerate(res, 1):
                print(f"{i}. {res['title']}")

        case "build":
            build_command()
        
        case "tf":
            tf_command(args.doc_id, args.term)
        
        case "idf":
            idf_command(args.term)

        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        
        case "bm25idf":
            bm25_idf_command(args.term)
        
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        
        case "bm25search":
            bm25search_command(args.query)


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()