import os,sys
from src.exceptions import ApplicationException 
from src.logger import logging
from datetime import datetime 
from src.utils.utils import read_yaml_file 
from src.constants import * 

config_data = read_yaml_file(CONFIG_FILE_PATH) 

class TrainingPipelineConfig:
    
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
            
            
        except Exception  as e:
            raise ApplicationException(e,sys)    


class DataIngestionConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            data_ingestion_key=config_data[DATA_INGESTION_CONFIG_KEY]
            
            
            self.database_name=data_ingestion_key[DATA_INGESTION_DATABASE_NAME]
            self.collection_name=data_ingestion_key[DATA_INGESTION_COLLECTION_NAME]
            
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir ,data_ingestion_key[DATA_INGESTION_ARTIFACT_DIR])
            self.raw_data_dir = os.path.join(self.data_ingestion_dir,data_ingestion_key[DATA_INGESTION_RAW_DATA_DIR_KEY])
            self.ingested_data_dir=os.path.join(self.raw_data_dir,data_ingestion_key[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            self.train_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TRAIN_DIR_KEY])
            self.test_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TEST_DIR_KEY])
            self.test_size = 0.2

        except Exception  as e:
            raise ApplicationException(e,sys) 
        
class DataValidationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:

            data_validation_key=config_data[DATA_VALIDATION_CONFIG_KEY]
            
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir ,data_validation_key[DATA_VALIDATION_ARTIFACT_DIR])
            self.validated_dir=os.path.join(training_pipeline_config.artifact_dir,data_validation_key[DATA_VALIDATION_VALID_DATASET])
            self.validated_train_path=os.path.join(self.data_validation_dir,data_validation_key[DATA_VALIDATION_TRAIN_FILE])
            self.validated_test_path=os.path.join(self.data_validation_dir,data_validation_key[DATA_VALIDATION_TEST_FILE])
            self.schema_file_path=SCHEMA_FILE_PATH

        except Exception as e:
            raise ApplicationException(e,sys) from e 
        
class DataTransformationConfig:
    
    try:

        def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            
            
            data_transformation_key=config_data[DATA_TRANSFORMATION_CONFIG_KEY]
            
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , data_transformation_key[DATA_TRANSFORMATION])
            self.transformation_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_DIR_NAME_KEY])
            self.transformed_train_dir = os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
            self.transformed_test_dir = os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])
            self.preprocessed_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])
            self.feature_engineering_object_file_path =os.path.join(self.preprocessed_dir,data_transformation_key[DATA_TRANSFORMATION_FEA_ENG_FILE_NAME_KEY])

    except Exception as e:
        raise ApplicationException(e,sys) from e 
    
    