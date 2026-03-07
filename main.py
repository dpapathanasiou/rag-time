from pathlib import Path
from rag import CORPUS_DIR, RAG_CHAIN, rebuild_index

def cli():
    """Simple command line interface"""
    print("Welcome! Let's talk, ask me a question\n(Ctrl+C to exit)")
    while True:
        try:
            q = input("\n> ")
            if not q.strip():
                continue
            answer = RAG_CHAIN.invoke(q)
            print(f"\n{answer}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    corpus = Path(CORPUS_DIR)
    if not corpus.exists():
        corpus.mkdir(parents=True, exist_ok=True)
    rebuild_index(corpus)

    cli()
