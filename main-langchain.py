import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, URL, text
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

load_dotenv()

db = chromadb.PersistentClient(path="./data")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

llm = Ollama(
    model="gemma:7b-instruct",
    temperature=0.1,
    request_timeout=600.0
)

embed_model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
)


def load_documents_from_db():
    # https://docs.sqlalchemy.org/en/20/core/engines.html#creating-urls-programmatically
    connection_url = URL.create(
        os.environ.get("DB_CONNECTION"),
        username=os.environ.get("DB_USERNAME"),
        password=os.environ.get("DB_PASSWORD"), # plain (unescaped) text
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_DATABASE"),
        port=os.environ.get("DB_PORT"),
        query={'charset': 'utf8mb4', 'collation': 'utf8mb4_unicode_ci'}
    )
    engine  = create_engine(
        connection_url,
        isolation_level="REPEATABLE READ", 
        echo=True
    )
    # connect to database server 
    with engine.connect() as connection:
        # connection = connection.execution_options(isolation_level="READ COMMITTED")
        #executing the query
        result = connection.execute(text("SELECT id,context FROM m_chat_ai_reply_context"))
        # commit to prevent rollback log
        connection.commit()

        return result

# def split_document_to_chunks(document):
    # Split the documents into sentences based on separators


def ask(question: str):
    loader = WebBaseLoader("https://www.abilive.vn/")
    data = loader.load()
    # print(data)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    # all_splits = text_splitter.split_text(data)
    # print(all_splits)
    # for index, row in enumerate(all_splits):
    #      print(f"Node {index + 1}: \n" , row)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=8192, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(data)
    print(doc_splits)

    # Create a vector store using Chroma DB, our chunked data from the URLs, and the nomic-embed-text embedding model
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    # splitter = SentenceSplitter(
    #     chunk_size=512,
    #     # chunk_overlap=20,
    # )
    
    # rows = load_documents_from_db()

    # for row in rows:
    #     chunks = splitter.get_nodes_from_documents(text=row.context)
    #     for index, chunk in enumerate(chunks):
    #         print(f"Node {index + 1}: \n" , chunk)
    #     print("Context:" , row.context)

     
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py '<question>'")
        sys.exit(1)

    question = sys.argv[1]
    answer = ask(question)
    # print(answer)