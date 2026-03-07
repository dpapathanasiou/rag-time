# About

This is a simple [Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) application, to complement any locally-running [Ollama](https://ollama.com/) model.

# Initial Setup

*Only needed if ever want to recreate from scratch, without using the current [uv.lock](uv.lock)* - create a [uv project](https://docs.astral.sh/uv/#project-structure) as follows:

```sh
uv init rag-time --python 3.13
uv add langchain langchain-core langchain-community langchain-ollama langchain-text-splitters langchain-chroma chromadb sentence-transformers pypdf beautifulsoup4
source .venv/bin/activate
```
