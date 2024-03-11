import pandas as pd
import json
import os
from dotenv import load_dotenv
import pymongo

def mongo_client():
    ROOT_DIR = os.getcwd()
    env_file_path = os.path.join(ROOT_DIR, '.env')

    # Load environment variables from .env file
    load_dotenv(dotenv_path=env_file_path)

    # username = os.getenv('USER_NAME')
    # password = os.getenv('PASS_WORD')
    # cluster_name = os.getenv('CLUSTER_LABEL')

    # Use the escaped username and password in the MongoDB connection string
    # mongo_db_url = f"mongodb+srv://{username}:{password}@{cluster_name}.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    mongo_db_url=os.getenv("MONGO_DB_URL")
    
    print(mongo_db_url)
    client = pymongo.MongoClient(mongo_db_url)
    
    return client
