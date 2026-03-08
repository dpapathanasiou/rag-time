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
    # TODO: initialize config based on args to main
    config = RAGConfig()
    vector_store = rebuild_index(config)
    rag_chain = create_rag_chain(config)
    cli(rag_chain)
