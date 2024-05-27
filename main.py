import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text
from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# load llm
# https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/
llm = ChatOllama(
    model="gemma:7b-instruct",
    temperature=0,
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
            documents.append(row.context)

    return documents

def ask():
    # Load documents from URLs
    urls = [
        "https://www.abilive.vn/"
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs_list)

    # create the open-source embedding function
    embedding = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    # load it into Chroma
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding,
        collection_name='website'
    )

    retriever = db.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 2}
    )

    # Create a question / answer pipeline 
    rag_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # question = 'Người đại diện công ty là ai ?'
    # question = '所在地 ベトナムオフィス ?'
    # question = 'location of Hokkaido Branch ?'
    # question = 'address of Hokkaido Branch ?'
    # question = '名古屋本社の所在地 ?'
    # question = 'アビリブの3つの特長 ?'
    # question = 'telephone number of Nagoya Head Office ?'
    question = 'introduce Abilive Vietnam ?'

    # Invoke the pipeline
    print(rag_chain.invoke(question))
     
if __name__ == "__main__":
    ask()
    # print(answer)