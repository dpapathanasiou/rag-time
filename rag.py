from os import getenv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SOURCE_CODE = [
    ".c",  # Language.C,
    ".h",  # Language.C,
    ".cbl",  # Language.COBOL,
    ".cob",  # Language.COBOL,
    ".cpy",  # Language.COBOL,
    ".cs",  # Language.CSHARP,
    ".cpp",  # Language.CPP,
    ".ex",  # Language.ELIXIR,
    ".exs",  # Language.ELIXIR,
    ".go",  # Language.GO,
    ".hs",  # Language.HASKELL,
    ".java",  # Language.JAVA,
    ".js",  # Language.JS,
    ".jsx",  # Language.JS,
    ".json",  # Language.JS,
    ".kt",  # Language.KOTLIN,
    ".lua",  # Language.LUA,
    ".php",  # Language.PHP,
    ".pl",  # Language.PERL,
    ".py",  # Language.PYTHON,
    ".r",  # Language.R,
    ".rst",  # Language.RST,
    ".rb",  # Language.RUBY,
    ".rs",  # Language.RUST,
    ".scala",  # Language.SCALA,
    ".swift",  # Language.SWIFT,
    ".tex",  # Language.LATEX,
    ".latex",  # Language.LATEX,
    ".ts",  # Language.TS,
    ".tsx",  # Language.TS,
]


class RAGConfig:
    def __init__(
        self,
        chunk_size=None,
        chunk_overlap=None,
        collection_name=None,
        retrieval_keys=None,
    ):
        self.base_model = getenv("BASE_MODEL", "gpt-oss")
        self.embed_model = getenv("EMBED_MODEL", "embeddinggemma")

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
        self.retrieval_keys = 4 if retrieval_keys is None else retrieval_keys
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.retrieval_keys}
        )

        self.chunk_size = 800 if chunk_size is None else chunk_size
        self.chunk_overlap = 100 if chunk_overlap is None else chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

    def get_base_model(self):
        return OllamaLLM(model=self.base_model)

    def get_embeddings(self):
        return OllamaEmbeddings(model=self.embed_model)

    def __str__(self):
        return f"""RAGConfig:
        CORPUS_DIR\t= {self.corpus_path.name}
        CHROMA_DIR\t= {self.chroma_path.name}

        Base Model\t= {self.base_model}
        Embed Model\t= {self.embed_model}

        chunk_size\t= {self.chunk_size}
        chunk_overlap\t= {self.chunk_overlap}
        collection_name\t= {self.collection_name}
        retrieval_keys\t= {self.retrieval_keys}
        """


def load_corpus(corpus_folder: Path):
    docs = []
    names = []
    for p in corpus_folder.glob("**/*"):
        file_ext = p.suffix.lower()
        if file_ext in SUPPORTED_SOURCE_CODE:
            names.append(f" - {p.name}")
            docs.extend(
                GenericLoader.from_filesystem(p, parser=LanguageParser()).load()
            )
        elif file_ext in {".pdf", ".css", ".htm", ".html", ".md", ".txt", ".text"}:
            names.append(f" - {p.name}")
            match file_ext:
                case ".pdf":
                    docs.extend(PyPDFLoader(p).load())
                case ".css" | ".htm" | ".html":
                    docs.extend(BSHTMLLoader(p).load())
                case ".md":
                    docs.extend(UnstructuredMarkdownLoader(p).load())
                case _:
                    docs.extend(TextLoader(p, autodetect_encoding=True).load())
    print(f"Corpus files:\n{chr(10).join(names)}\n")
    return docs


def rebuild_index(config: RAGConfig, force=False):
    vector_store = config.vector_store

    index = vector_store.get()
    if not force:
        if index["ids"]:
            print("Corpus already exists and is not empty, skipping rebuild")
            return

    text_splitter = config.text_splitter
    # TODO: use a more appropriate splitter by doc type (https://docs.langchain.com/oss/python/integrations/document_loaders/source_code#splitting)
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
