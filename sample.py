import mysql.connector
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

llm = Ollama(model="gemma:7b-instruct", request_timeout=600.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

def fetch_data_from_mysql(host, user, password, database, query):
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=3307
    )
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data

def build_index_from_data(data):
    text_list = [item[0] for item in data]
    documents = [Document(text=t) for t in text_list]
    index = VectorStoreIndex.from_documents(documents)

    return index

def generate_response(question, context):
    # Concatenate question and context
    input_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    # Assuming response['choices'][0]['text'] contains the answer
    answer = llm.complete(input_text)

    return answer

def main():
    # Database connection parameters
    host = '127.0.0.1'
    user = 'root'
    password = ''
    database = 'abichat_dev_tamada'
    query = 'SELECT context FROM m_chat_ai_reply_context'

    # Fetch data from MySQL
    data = fetch_data_from_mysql(host, user, password, database, query)

    # Build the index
    index = build_index_from_data(data)
    query_engine = index.as_query_engine()
    print(index)
    context = query_engine.query('Viet Nam office location?')
    print(generate_response('Viet Nam office location?', context))

#     print("Welcome to the Q&A chat app! Type 'exit' to quit.")
#     while True:
#         question = input("You: ")
#         if question.lower() == 'exit':
#             break
#         context = query_engine.query(question)
#         answer = generate_response(question, context)
#         print("Bot:", answer)

if __name__ == "__main__":
    main()