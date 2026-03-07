from os import getenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings

CORPUS_DIR = getenv("CORPUS_DIR", "/tmp")
CHROMA_DIR = getenv("CHROMA_DIR", "/tmp")

BASE_MODEL = OllamaLLM(model=getenv("BASE_MODEL", "gpt-oss"))
EMBEDDINGS = OllamaEmbeddings(model=getenv("EMBED_MODEL", "embeddinggemma"))

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100, 
    add_start_index=True
)

def load_corpus(corpus_folder):
    docs = []
    names = []
    for p in corpus_folder.glob("**/*"):
        if p.suffix.lower() == ".pdf":
            names.append(p.name)
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in {".md", ".txt", ".text"}:
            names.append(p.name)
            docs.extend(TextLoader(str(p), autodetect_encoding=True).load())

    print(f"Corpus files:\n{chr(10).join(names)}")
    return docs

def rebuild_index(corpus_folder, text_splitter=None, vector_store=None):
    if text_splitter is None:
        text_splitter = TEXT_SPLITTER

    if vector_store is None:
        vector_store = VECTOR_STORE
    
    # TODO: have vector_store detect if prior index has been persisted, and skip this

    data = text_splitter.split_documents(load_corpus(corpus_folder))

    vector_store.reset_collection()
    vector_store.add_documents(data)

VECTOR_STORE = Chroma(
    collection_name="local_rag_corpus", 
    embedding_function=EMBEDDINGS, 
    persist_directory=CHROMA_DIR
)
RETRIEVER = VECTOR_STORE.as_retriever(search_kwargs={"k": 4})

PROMPT = ChatPromptTemplate.from_template(
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

RAG_CHAIN = (
    {"context": RETRIEVER, "question": RunnablePassthrough()}
    | PROMPT
    | BASE_MODEL
    | StrOutputParser()
)
