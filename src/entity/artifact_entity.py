from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validated_train_path:str
    validated_test_path:str