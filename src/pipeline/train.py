import uuid 
import sys 
from src.exceptions import ApplicationException
from typing import List
from multiprocessing import Process 
from src.entity.config_entity import * 
from src.entity.artifact_entity import * 
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


class Pipeline():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config

            
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(self.training_pipeline_config))
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=DataValidationConfig(self.training_pipeline_config),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    
    def run_pipeline(self):
        try:
             #data ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

        except Exception as e:
            raise ApplicationException(e,sys) from e 