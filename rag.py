from collections import defaultdict
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
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

TEXT = [".pdf", ".css", ".htm", ".html", ".md", ".txt", ".text"]

SOURCE_CODE = {
    ".c": Language.C,
    ".h": Language.C,
    ".cbl": Language.COBOL,
    ".cob": Language.COBOL,
    ".cpy": Language.COBOL,
    ".cs": Language.CSHARP,
    ".cpp": Language.CPP,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    ".go": Language.GO,
    ".hs": Language.HASKELL,
    ".java": Language.JAVA,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".json": Language.JS,
    ".kt": Language.KOTLIN,
    ".lua": Language.LUA,
    ".php": Language.PHP,
    ".pl": Language.PERL,
    ".py": Language.PYTHON,
    ".r": Language.R,
    ".rst": Language.RST,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".tex": Language.LATEX,
    ".latex": Language.LATEX,
    ".ts": Language.TS,
    ".tsx": Language.TS,
}


class RAGConfig:
    def __init__(
        self,
        chunk_size=None,
        chunk_overlap=None,
        collection_name=None,
        retrieval_keys=None,
        base_prompt=None,
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

        if base_prompt is None:
            prompt_path = Path.cwd() / "prompts" / "default_prompt.txt"
            self.base_prompt = prompt_path.read_text()
        else:
            prompt_path = Path(base_prompt)
            self.base_prompt = prompt_path.read_text()

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

        base_prompt\t= `{self.base_prompt[:50]}` ...
        """


def load_corpus(corpus_folder: Path):
    docs = defaultdict(list)  # k=file extension, v=list of loaded docs
    corpus_files = []

    for p in corpus_folder.glob("**/*"):
        file_ext = p.suffix.lower()
        if file_ext in SOURCE_CODE.keys():
            corpus_files.append(f" - {p.name}")
            docs[file_ext].extend(
                GenericLoader.from_filesystem(p, parser=LanguageParser()).load()
            )
        elif file_ext in TEXT:
            corpus_files.append(f" - {p.name}")
            match file_ext:
                case ".pdf":
                    docs[file_ext].extend(PyPDFLoader(p).load())
                case ".css" | ".htm" | ".html":
                    docs[file_ext].extend(BSHTMLLoader(p).load())
                case ".md":
                    docs[file_ext].extend(UnstructuredMarkdownLoader(p).load())
                case _:
                    docs[file_ext].extend(
                        TextLoader(p, autodetect_encoding=True).load()
                    )
    print(f"Corpus files:\n{chr(10).join(corpus_files)}\n")

    return dict(docs)


def rebuild_index(config: RAGConfig, force=False):
    vector_store = config.vector_store

    index = vector_store.get()
    if not force:
        if index["ids"]:
            print("Corpus already exists and is not empty, skipping rebuild")
            return

    vector_store.reset_collection()

    docs_by_language = load_corpus(config.corpus_path)
    for extension, docs in docs_by_language.items():
        if extension in SOURCE_CODE.keys():
            language = SOURCE_CODE[extension]
            print(f"- indexing {len(docs)} docs for {language} ({extension})")
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=SOURCE_CODE[extension],
                chunk_size=config.chunk_size,
                chunk_overlap=0,  # confusing in the case of source code
                add_start_index=True,
            )
            data = splitter.split_documents(docs)
            vector_store.add_documents(data)
        elif extension in TEXT:
            print(f"- indexing {len(docs)} docs as text ({extension})")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                add_start_index=True,
            )
            data = splitter.split_documents(docs)
            vector_store.add_documents(data)


def create_rag_chain(config: RAGConfig):
    prompt = ChatPromptTemplate.from_template(
        config.base_prompt
        + """

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
