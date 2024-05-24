import os
import sys
from typing import List
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage
# from llama_index.embeddings.instructor import InstructorEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
import chromadb
from llama_index.core.schema import TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from chromadb import EmbeddingFunction, Documents, Embeddings
from llama_index.readers.database import DatabaseReader

load_dotenv()

class LlamaIndexEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, ef: BaseEmbedding):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        return [node.embedding for node in self.ef([TextNode(text=doc) for doc in input])]

Settings.chunk_size=1024
Settings.chunk_overlap=20

# load llm
# https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/
llm = Ollama(
    model="gemma:7b-instruct",
    temperature=0,
    request_timeout=600.0
)
Settings.llm = llm

# define embedding model
# https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # model_name="intfloat/multilingual-e5-large-instruct"
)
Settings.embed_model = embed_model

# initialize client, setting path to save data
# https://docs.llamaindex.ai/en/stable/understanding/storing/storing/
db = chromadb.PersistentClient(path="./.data/chroma_db")
# create or load collection
collection_name = "quickstart"
chroma_collection = db.get_or_create_collection(
    collection_name, 
    embedding_function=LlamaIndexEmbeddingAdapter(embed_model)
)
# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

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

    query = "SELECT id,context FROM m_chat_ai_reply_context"
    
    documents = []
    # connect to database server 
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
    with engine.connect() as connection:
        # connection = connection.execution_options(isolation_level="READ COMMITTED")
        #executing the query
        result = connection.execute(text(query))
        # commit to prevent rollback log
        connection.commit()

        for row in result:
            documents.append(Document(text=row.context))

    return documents

    # reader = DatabaseReader(
    #     engine=engine
    # )

    # documents = reader.load_data(query=query)
    # return documents
    

def creat_document_from_text(text):
    return Document(text=text)

def load_documents_from_urls(urls: List[str]):
    markdown_docs = SimpleWebPageReader(html_to_text=True).load_data(urls)
    return markdown_docs

def split_documents_to_chunks(documents):
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

def index_documents():
    # load documents
    # https://docs.llamaindex.ai/en/stable/understanding/loading/loading/
    # documents = load_documents_from_urls(["https://www.abilive.vn/"])
    # text_splitter = MarkdownNodeParser()

    documents = load_documents_from_db()
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    # indexing documents
    # https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing/
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context, 
        embed_model=embed_model,
        transformations=[text_splitter]
    )

    # Query Data
    # https://docs.llamaindex.ai/en/stable/understanding/querying/querying/
    # query_engine = vector_index.as_query_engine()
    # response = query_engine.query("Công ty làm về lĩnh vực gì ?")
    # print(response)

def test_llm():
    response = llm.complete("Hồ Chí Minh là ai ?")
    print(response)

def ask():
    # load your index from stored vectors
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    query_engine = vector_index.as_query_engine()
    response = query_engine.query("Công ty làm về lĩnh vực gì ?")
    print(response)
    response = query_engine.query("Người đại diện công ty là ai ?")
    print(response)
    response = query_engine.query("Công ty có chi nhánh tại việt nam không ?")
    print(response)
    response = query_engine.query("Abilive Viet Nam Co., Ltd location ?")
    print(response)
     
if __name__ == "__main__":
    if sys.argv[1] == 'index':
        index_documents()
        sys.exit(1)

    answer = ask()
    # print(answer)