import argparse
from utils.commands import verify_command, embed_text_command, verify_embeddings_command, embed_query_command, semantic_search, chunk_command, semantic_chunk

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    subparser.add_parser("verify", help="Verify embedding model")
    subparser.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embeder = subparser.add_parser("embed_text", help="Embed text")
    query_embeder = subparser.add_parser("embed_query", help="embed query")

    embeder.add_argument("text", type=str, help="Text to be embedded")
    query_embeder.add_argument("text", type=str, help="Text to be embedded")

    searcher = subparser.add_parser("search", help="embed query")
    searcher.add_argument("query", type=str, help="query")
    searcher.add_argument("--limit", type=int, help="limit")

    chunker = subparser.add_parser("chunk", help="chunk text")
    chunker.add_argument("text", type=str, help="text")
    chunker.add_argument("--chunk-size", type=int, default=200, help="chunk-size")
    chunker.add_argument("--overlap", type=int, help="overlap", default=0)

    sem_chunk = subparser.add_parser("semantic_chunk", help="chunk text")
    sem_chunk.add_argument("text", type=str, help="text")
    sem_chunk.add_argument("--max-chunk-size", type=int, default=4, help="chunk-size")
    sem_chunk.add_argument("--overlap", type=int, help="overlap", default=0)

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
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)


        
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()