import os
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text

load_dotenv()

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
    #executing the query
    result = connection.execute(text("SELECT * FROM m_chat_ai_reply_context"))
    # commit to prevent rollback log
    connection.commit() 
    for row in result:
        print("Context:" , row.context)

# comment this 