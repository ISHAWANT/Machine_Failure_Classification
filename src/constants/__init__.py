import os

FILENAME='data.csv'

# Schema File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='schema.yaml'
SCHEMA_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)

# Config File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)

# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DATABASE_NAME= "data_base"
DATA_INGESTION_COLLECTION_NAME= "collection_name"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"
CONFIG_FILE_KEY = "config"

# Data Validation related variable
DATA_VALIDATION_ARTIFACT_DIR="data_validation_dir"
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_VALID_DATASET ="validated_data"
DATA_VALIDATION_TRAIN_FILE = "Train_data"
DATA_VALIDATION_TEST_FILE ="Test_data"