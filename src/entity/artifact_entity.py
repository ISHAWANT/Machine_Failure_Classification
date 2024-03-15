from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validated_train_path:str
    validated_test_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    feature_engineering_object_file_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    model_artifact_report:str

@dataclass
class ModelEvaluationArtifact:
    model_name:str
    Auc_Score:float
    selected_model_path:str
    model_report_path:str  

@dataclass
class ModelPusherArtifact:
    message:str