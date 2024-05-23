import os
import sys
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, URL, text
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
import chromadb

load_dotenv()

db = chromadb.PersistentClient(path="./data")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm = Ollama(
    model="gemma:7b-instruct",
    temperature=0.1,
    request_timeout=600.0
)

# https://huggingface.co/intfloat/multilingual-e5-large-instruct
embed_model = SentenceTransformer(
    # 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    'intfloat/multilingual-e5-large-instruct'
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

# https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
def creat_document_from_text(text):
    return Document(text=text)
    
def load_documents_from_urls(urls: List[str]):
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    return documents

def split_documents_to_chunks(documents):
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

# def embed_chunks(chunks):
    

def index_documents():
    documents = load_documents_from_urls(["https://www.abilive.vn/"])
    chunks = split_documents_to_chunks(documents)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context, 
        embed_model=embed_model
    )

    # for index, row in enumerate(chunks):
    #      print(f"Node {index + 1}: \n" , row)

def ask():
    resp = llm.complete("Who is Paul Graham?")
    print(resp)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    # all_splits = text_splitter.split_text(data)
    # print(all_splits)
    # for index, row in enumerate(all_splits):
    #      print(f"Node {index + 1}: \n" , row)

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=8192, chunk_overlap=100)
    # doc_splits = text_splitter.split_documents(data)
    # print(doc_splits)

    # Create a vector store using Chroma DB, our chunked data from the URLs, and the nomic-embed-text embedding model
    # vectorstore = Chroma.from_documents(
    #     documents=doc_splits,
    #     collection_name="rag-chroma",
    #     embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    # )
    # retriever = vectorstore.as_retriever()

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
    if sys.argv[1] == 'index':
        print("Indexing documents")
        index_documents()
        sys.exit(1)

    answer = ask()
    # print(answer)