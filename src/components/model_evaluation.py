
from src.exceptions import ApplicationException
from src.logger import logging
import sys
import os
from src.utils import *
from src.entity.config_entity import *
from src.entity.artifact_entity import *
from src.constants import *

from src.utils.utils import read_yaml_file,write_yaml_file

class ModelEvaluation:


    def __init__(self,model_evaluation_config:ModelEvalConfig,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:

    
            self.model_trainer_artifact=model_trainer_artifact
            self.model_evaluation_config=model_evaluation_config

            self.saved_model_config=SavedModelConfig()
            
            self.saved_model_directory=self.saved_model_config.saved_model_dir
            
        except Exception as e:
            raise ApplicationException(e,sys)
        
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(" Model Evaluation Started ")
            ## Artifact trained Model  files
            model_trained_artifact_path = self.model_trainer_artifact.trained_model_file_path
            model_trained_report = self.model_trainer_artifact.model_artifact_report
            
            # Saved Model files

            saved_model_path = self.saved_model_config.saved_model_file_path
            saved_model_report_path=self.saved_model_config.saved_model_report_path

                        
            # Loading the models
            logging.info("Saved_models directory .....")
            os.makedirs(self.saved_model_directory,exist_ok=True)
            
            # Check if SAVED_MODEL_DIRECTORY is empty
            if not os.listdir(self.saved_model_directory):
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                artifact_model_r2_score =float( model_trained_report_data['Auc_Score'])
                model_name = model_trained_report_data['Model_name']
                Auc_Score = artifact_model_r2_score
                # Artifact ----> Model, Model Report 
                model_path = model_trained_artifact_path
                model_report_path = model_trained_report
                
                comment='Trained model Saved '
                
            else:
                saved_model_report_data = read_yaml_file(file_path=saved_model_report_path)
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
        

                # Compare the AUC Score and accuracy of the two models
                saved_model_score = float(saved_model_report_data['Auc_Score'])

                artifact_model_score =float(model_trained_report_data['Auc_Score'])

                # Compare the models and log the result
                if artifact_model_score > saved_model_score:
                    logging.info("Trained model outperforms the saved model!")
                    model_path = model_trained_artifact_path
                    
                    model_report_path = model_trained_report
                    model_name = model_trained_report_data['Model_name']
                    Auc_Score = float( model_trained_report_data['Auc_Score'])
                    
                    comment='Trained Model is better than Saved model'
                    
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"Auc_Score : {Auc_Score}")
                  
                elif artifact_model_score < saved_model_score:
                    logging.info("Saved model outperforms the trained model!")
                    model_path = saved_model_path
                    comment='Saved Model is better than Currently trained model'
                    model_report_path = saved_model_report_path
                    model_name = saved_model_report_data['Model_name']
                    Auc_Score = float( saved_model_report_data['Auc_Score'])
                    logging.info(f"Model Selected : {model_name}")

                    logging.info(f"Auc_Score : {Auc_Score}")
            
                else:
                    logging.info("Both models have the same Auc_Score.")
                    model_path = saved_model_path
                    comment='Saved Model is better than Currently trained model'
                    model_report_path = saved_model_report_path
                    model_name = saved_model_report_data['Model_name']
                    Auc_Score = float( saved_model_report_data['Auc_Score'])
                    logging.info(f"Model Selected : {model_name}")

                    logging.info(f"Auc_Score : {Auc_Score}")
                    
                    
            model_eval_report= {'Model_name':  model_name,
                                'Auc_Score': Auc_Score,
                                'Message': comment}
            
            write_yaml_file(file_path=self.model_evaluation_config.model_eval_report,data=model_eval_report)
            

            # Create a model evaluation artifact
            model_evaluation = ModelEvaluationArtifact(model_name=model_name,Auc_Score=Auc_Score,
                                                    selected_model_path=model_path, 
                                                    model_report_path=model_report_path)

            logging.info("Model evaluation completed successfully!")

            return model_evaluation
        except Exception as e:
            logging.error("Error occurred during model evaluation!")
            raise ApplicationException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")