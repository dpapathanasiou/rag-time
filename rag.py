from os import getenv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGConfig:
    def __init__(
        self,
        chunk_size=None,
        chunk_overlap=None,
        collection_name=None,
        retrieval_keys=None,
    ):
        self.corpus_path = Path(getenv("CORPUS_DIR", ".corpus"))
        if not self.corpus_path.exists():
            self.corpus_path.mkdir(parents=True, exist_ok=True)

        self.chroma_dir = getenv("CHROMA_DIR", ".chroma_db")
        self.chroma_path = Path(self.chroma_dir)
        if not self.chroma_path.exists():
            self.chroma_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = (
            "local_corpus" if collection_name is None else collection_name
        )
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.get_embeddings(),
            persist_directory=self.chroma_dir,
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4 if retrieval_keys is None else retrieval_keys}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800 if chunk_size is None else chunk_size,
            chunk_overlap=100 if chunk_overlap is None else chunk_overlap,
            add_start_index=True,
        )

    def get_base_model(self):
        return OllamaLLM(model=getenv("BASE_MODEL", "gpt-oss"))

    def get_embeddings(self):
        return OllamaEmbeddings(model=getenv("EMBED_MODEL", "embeddinggemma"))


def load_corpus(corpus_folder: Path):
    docs = []
    names = []
    for p in corpus_folder.glob("**/*"):
        file_ext = p.suffix.lower()
        if file_ext in {".pdf", ".md", ".txt", ".text"}:
            names.append(f" - {p.name}")
            if file_ext == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            else:
                docs.extend(TextLoader(str(p), autodetect_encoding=True).load())

    print(f"Corpus files:\n{chr(10).join(names)}\n")
    return docs


def rebuild_index(config: RAGConfig):
    text_splitter = config.text_splitter
    vector_store = config.vector_store

    # TODO: have vector_store detect if prior index has been persisted, and skip this

    data = text_splitter.split_documents(load_corpus(config.corpus_path))

    vector_store.reset_collection()
    vector_store.add_documents(data)


def create_rag_chain(config: RAGConfig):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an efficient, diligent, and helpful assistant.  Please
        attempt to answer the user's question here, using the context
        provided. If the answer is not in the context, please just say you
        don't know.

        Context:
        {context}

        Question: {question}
        """
    )

    return (
        {"context": config.retriever, "question": RunnablePassthrough()}
        | prompt
        | config.get_base_model()
        | StrOutputParser()
    )
