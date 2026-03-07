# About

This is a simple [Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) application, to complement any locally-running [Ollama](https://ollama.com/) model.

# Initial Setup

*Only needed if ever want to recreate from scratch, without using the current [uv.lock](uv.lock)* - create a [uv project](https://docs.astral.sh/uv/#project-structure) as follows:

```sh
uv init rag-time --python 3.13
uv add langchain \
  langchain-core \
  langchain-community \
  langchain-ollama \
  langchain-text-splitters \
  langchain-chroma \
  chromadb \
  sentence-transformers \
  pypdf \
  beautifulsoup4 \
  chardet
source .venv/bin/activate
```

Prepare the document corpus and chroma folders:

```sh
mkdir -p "$(pwd)/.corpus"
mkdir -p "$(pwd)/.chroma_db"
```

# Usage

Define the enviroment variables consistent with the current setup:

```sh
export CORPUS_DIR=".corpus"
export CHROMA_DIR=".chroma_db"
export BASE_MODEL="gpt-oss"
export EMBED_MODEL="embeddinggemma"
```

Copy the RAG corpus documents into the folder defined by `$CORPUS_DIR`, and start the cli:

```sh
uv run main.py
Corpus files:
    ...[listed line by line here, text and pdf]...
Welcome! Let's talk, ask me a question
(Ctrl+C to exit)

> 
```