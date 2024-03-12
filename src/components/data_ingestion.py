import os,sys
from six.moves import urllib 
from src.constants import * 
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact 
from src.utils.utils import get_collection_as_dataframe 
from src.logger import logging 
from src.exceptions import ApplicationException 
import pandas as pd 
import shutil 
from sklearn.model_selection import train_test_split 

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>'*30}Data Ingestion log started.{'<<'*30} \n\n")
            self.data_ingestion_config = data_ingestion_config
            

        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    
    def get_data_from_mongo(self):
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)

            logging.info("Saving Data from Database to local folder ....")
            
            # Raw Data Directory Path
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            logging.info(f" Raw Data directory : {raw_data_dir}")

            # Make Raw data Directory
            os.makedirs(raw_data_dir, exist_ok=True)
            
            csv_file_name='train.csv'
            raw_file_path = os.path.join(raw_data_dir, csv_file_name)
            df.to_csv(raw_file_path)
            
        
            # copy the the extracted csv from raw_data_dir ---> ingested Data 
            ingest_directory=os.path.join(self.data_ingestion_config.ingested_data_dir)
            os.makedirs(ingest_directory,exist_ok=True)

            # Updating file name 
            ingest_file_path = os.path.join(self.data_ingestion_config.ingested_data_dir, csv_file_name)
            
            # Copy the extracted CSV file
            shutil.copy2(raw_file_path, ingest_file_path)
            
            logging.info(" Data stored in ingested Directory ")
            

            
            return ingest_file_path
            

        except Exception as e:
            raise ApplicationException(e, sys) from e 
        
        
        
    def split_csv_to_train_test(self,csv_file_path):
        
                    
        train_file_path=self.data_ingestion_config.train_file_path
        test_file_path=self.data_ingestion_config.test_file_path
        
        os.makedirs(train_file_path)
        os.makedirs(test_file_path)
        
        
        # Load data from the CSV file
        data = pd.read_csv(csv_file_path,index_col=0)

        size=self.data_ingestion_config.test_size
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(data, test_size=size, random_state=42)

        # Save the training and testing data into separate CSV files
        train_file_path=os.path.join(train_file_path,FILENAME)
        test_file_path=os.path.join(test_file_path,FILENAME)
        
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        
        logging.info(f" Train File path : {train_file_path}")
        logging.info(f" Test File path : {test_file_path}")
        
        
        
        
        data_ingestion_artifact=DataIngestionArtifact(train_file_path=train_file_path,test_file_path=test_file_path)
        
        return data_ingestion_artifact



    def initiate_data_ingestion(self):
        try:
            
            logging.info("Donwloading data from mongo ")
            ingest_file_path=self.get_data_from_mongo()
            
            logging.info("Splitting data .... ")
            
            return  self.split_csv_to_train_test(csv_file_path=ingest_file_path)


        except Exception as e:
            raise ApplicationException(e,sys) from e