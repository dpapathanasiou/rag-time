import argparse

from rag import RAGConfig, create_rag_chain, rebuild_index


def cli(rag_chain):
    """Simple command line interface"""
    print("Welcome! Let's talk, ask me a question\n(Ctrl+C to exit)")
    while True:
        try:
            q = input("\n> ")
            if not q.strip():
                continue
            answer = rag_chain.invoke(q)
            print(f"\n{answer}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_size", help="number of characters per chunk", type=int, default=800
    )
    parser.add_argument(
        "--chunk_overlap",
        help="size of overlap between chunks in order to maintain context",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--collection_name",
        help="name of the corpus in the vector store",
        type=str,
        default="local_corpus",
    )
    parser.add_argument(
        "--retrieval_keys",
        help="number (max) of revelant chunks to retrieve",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--force_index_rebuild",
        help="rebuild the corpus index, regardless of whether or not it already exists",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--prompt",
        help="path to the file containing the base prompt (as plain text)",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    config = RAGConfig(
        args.chunk_size,
        args.chunk_overlap,
        args.collection_name,
        args.retrieval_keys,
        args.prompt,
    )
    print(config)
    vector_store = rebuild_index(config, force=args.force_index_rebuild)
    rag_chain = create_rag_chain(config)
    cli(rag_chain)
