import os
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text

load_dotenv()

# https://docs.sqlalchemy.org/en/20/core/engines.html#creating-urls-programmatically
con_url = URL.create(
    os.environ.get("DB_CONNECTION"),
    username=os.environ.get("DB_USERNAME"),
    password=os.environ.get("DB_PASSWORD"), # plain (unescaped) text
    host=os.environ.get("DB_HOST"),
    database=os.environ.get("DB_DATABASE"),
    port=os.environ.get("DB_PORT"),
    query={'charset': 'utf8mb4'}
)
engine  = create_engine(con_url, echo=True)
# connect to database server 
with engine.connect() as connection:
    #executing the query
    result = connection.execute(text("SELECT * FROM m_chat_ai_reply_context"))
    # print(result.fetchall())
    for row in result.mappings():
        print("Context:" , row["context"])

# comment this 