import os
import requests
from dotenv import load_dotenv

load_dotenv()


response = requests.get('https://httpbin.org/ip')
print('Your IP is {0}'.format(response.json()['origin']))
if 5 > 6:
    print("Five is greater than two!")
print("Five is greater than two!")
db = os.environ.get("DB_DATABASE")
print('Your DB is {0}'.format(db))
# comment this 