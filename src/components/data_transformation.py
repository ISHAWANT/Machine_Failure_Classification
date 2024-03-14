import os 
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exceptions import ApplicationException
from src.entity.artifact_entity import *
from src.entity.config_entity import *
from src.utils.utils import read_yaml_file,save_data,save_object
from src.constants import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,numerical_columns,categorical_columns,
                 target_columns,boolean_columns,drop_columns):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*10} Feature Engneering Started {'*'*10}\n\n")
        
        #Accesssing Column Labels 
                                
        #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              
                                    
        self.numerical_columns = numerical_columns
        self.categorical_columns=categorical_columns
        self.target_columns = target_columns
        self.boolean_columns=boolean_columns
        self.drop_columns=drop_columns
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")
    # Feature Engineering Pipeline 
    ######################### Data Modification ############################
    def clean_column_labels(self,dataframe):
        # Define the regular expression pattern to match special characters
        pattern = r'[^\w\s]'
        
        # Get the current column labels from the DataFrame
        column_labels = dataframe.columns.tolist()
        
        # Initialize a dictionary to store the mapping of old column labels to new ones
        column_mapping = {}
        
        # Iterate through each column label and clean it
        for label in column_labels:
            cleaned_label = label.replace(' ', '_')
            cleaned_label = re.sub(pattern, '', cleaned_label)
            column_mapping[label] = cleaned_label
        
        # Use the rename method to update the column labels in the DataFrame
        dataframe.rename(columns=column_mapping, inplace=True)
        
        return dataframe
    
    def drop_rows_with_nan(self, X: pd.DataFrame):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        
        # Drop rows with NaN values
        X = X.dropna()
        #X.to_csv("Nan_values_removed.csv", index=False)
        
        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        
        logging.info("Dropped NaN values.")
        
        return X

    def create_features(self,df):
        
        # Create a new feature by divided 'Air temperature' from 'Process temperature'
        df["Temperature_ratio"] = df['Process_temperature_K'] / df['Air_temperature_K']
        
        # Create a new feature by multiplying 'Torque' and 'Rotational speed'
        df['Torque * Rotational_speed'] = df['Torque_Nm'] * df['Rotational_speed_rpm']

        # Create a new feature by multiplying 'Torque' by 'Tool wear'
        df['Torque * Tool wear'] = df['Torque_Nm'] * df['Tool_wear_min']

        # Create a new feature by multiplying 'Torque' by 'Rotational speed'
        df['Torque * Rotational_speed'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
        
        return df


    def run_data_modification(self,data):
    
        X=data.copy()
        
        try:
            # Dropping unnecessary columns 
            X.drop(columns=self.drop_columns, axis=1, inplace=True)
        except Exception as e:
            # Handle the exception here, you can print an error message or take other appropriate actions.
            print("Columns not found :", e)
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)

        # Removing duplicated rows 
        X=self.clean_column_labels(X)
        
        # Modifying datatype 
        X=self.create_features(X)
        
        return X
    
    def encode_column(self,df, column_label):
        type_mapping = {
            'M': 0,
            'L': 1,
            'H': 2
        }
        
        # Check if the provided column label exists in the DataFrame
        if column_label not in df.columns:
            raise ValueError(f"Column '{column_label}' not found in the DataFrame.")
        
        # Create a new DataFrame with the encoded values
        new_df = df.copy()
        new_df[column_label] = new_df[column_label].map(type_mapping)
        
        return new_df

    def data_wrangling(self,X:pd.DataFrame):
        try:

            # Data Modification 
            data_modified=self.run_data_modification(data=X)
            
            logging.info(" Data Modification Done")
            # Map Encoding 
            data_encoded=self.encode_column(df=data_modified,column_label='Type')

            return data_encoded
    
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified=self.data_wrangling(X)

            #data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")
            
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shapde Modified Data : {data_modified.shape}")

            return data_modified
        except Exception as e:
            raise ApplicationException(e,sys) from e

class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
                                ############### Accesssing Column Labels #########################
                            
            # Reading data in Schema 
            self.transformation_yaml = read_yaml_file(file_path=TRANFORMATION_YAML_FILE_PATH)
            
            # Column data accessed from Schema.yaml
            self.target_column_name = self.transformation_yaml[TARGET_COLUMN_KEY]
            self.numerical_columns = self.transformation_yaml[NUMERICAL_COLUMN_KEY] 
            self.categorical_columns=self.transformation_yaml[CATEGORICAL_COLUMNS]
            self.boolean_columns=self.transformation_yaml[BOOLEAN_COLUMN]
            self.drop_columns=self.transformation_yaml[DROP_COLUMNS]
            
            # Tranformation 
            self.scaling_columns=self.transformation_yaml[SCALING_COLUMNS]
            
        except Exception as e:
            raise ApplicationException(e,sys) from e


    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(  numerical_columns=self.numerical_columns,
                                                                                categorical_columns=self.categorical_columns,
                                                                                boolean_columns=self.boolean_columns,
                                                                                target_columns=self.target_column_name,
                                                                                drop_columns=self.drop_columns
                                                                            ))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def initiate_data_transformation(self):
        try:
            # Data validation Artifact ------>Accessing train and test files 
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path
            test_file_path = self.data_validation_artifact.validated_test_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            
            logging.info(f" Accessing train file from :{train_file_path}\
                             Test File Path:{test_file_path} ")      
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            

            logging.info(f" Traning columns {train_df.columns}")
            
            # Schema.yaml ---> Extracting target column name
            target_column_name = self.target_column_name
            numerical_columns = self.numerical_columns
            categorical_columns=self.categorical_columns
            bool_columns=self.boolean_columns
                        
            # Log column information
            logging.info("Numerical columns: {}".format(numerical_columns))
            logging.info("Categorical columns: {}".format(categorical_columns))
            logging.info("Target Column: {}".format(target_column_name))
            logging.info("Boolean Column: {}".format(bool_columns))
            
            col = numerical_columns + categorical_columns+target_column_name + bool_columns
            # All columns 
            logging.info("All columns: {}".format(col))
            
            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 10 + " Training data " + "<<<" * 10)
            logging.info(f"Feature Enineering - Train Data ")
            train_df = fe_obj.fit_transform(X=train_df)
            logging.info(">>>" * 10 + " Test data " + "<<<" * 10)
            logging.info(f"Feature Enineering - Test Data ")
            test_df = fe_obj.transform(X=test_df)
            
            # Converting featured engineered array into dataframe
            # Train Data 
            logging.info(f"Feature Engineering of train and test Completed.")
            feature_eng_train_df:pd.DataFrame = train_df.copy()
          #  feature_eng_train_df.to_csv("feature_eng_train_df.csv")
            logging.info(f" Columns in feature enginering Train {feature_eng_train_df.columns}")
            logging.info(f"Feature Engineering - Train Completed")
            
            # Test Data
            feature_eng_test_df:pd.DataFrame = test_df.copy()
           # feature_eng_test_df.to_csv("feature_eng_test_df.csv")
            logging.info(f" Columns in feature enginering test {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training and testing dataframe.")
            
        
            # Train and Test Dataframe
            target_column_name=self.target_column_name
            input_feature_train_df = feature_eng_train_df.drop(columns=target_column_name,axis=1)
            train_target_df=feature_eng_train_df[target_column_name]
            input_feature_test_df= feature_eng_test_df.drop(columns=target_column_name,axis=1)
            test_target_df=feature_eng_test_df[target_column_name]
                        
            ############ Input Fatures transformation########
            ## Preprocessing 
            logging.info("*" * 10 + " Applying preprocessing object on training dataframe and testing dataframe " + "*" * 10)

            numerical_columns=self.scaling_columns
            
            logging.info(f" Scaling Columns : {numerical_columns}")
            
            logging.info("------- Before Transformed Data -----------")
            
            logging.info(f"Shape of Train Data : {input_feature_train_df.shape}")
            # Log the shape of Transformed Test
            logging.info(f"Shape of Test Data: {input_feature_test_df.shape}")
            
            scaler = MinMaxScaler()
            
            input_feature_train_df[numerical_columns] = scaler.fit_transform(input_feature_train_df[numerical_columns])
            input_feature_test_df[numerical_columns] = scaler.fit_transform(input_feature_test_df[numerical_columns])
            
            col_order=['Type']+ numerical_columns+ bool_columns
            input_feature_train_df=input_feature_train_df[col_order]
            input_feature_test_df=input_feature_test_df[col_order]
            
            logging.info(f" Transfromed Dataframe columns : {input_feature_train_df.columns}")
            
            # Log the shape of Transformed Train
            
            logging.info("------- Transformed Data -----------")
            
            logging.info(f"Shape of Train Data : {input_feature_train_df.shape}")
            # Log the shape of Transformed Test
            logging.info(f"Shape of Test Data: {input_feature_test_df.shape}")
            logging.info("Transformation completed successfully")
            
            train_data=pd.concat([input_feature_train_df,train_target_df],axis=1)
            test_data=pd.concat([input_feature_test_df,test_target_df],axis=1)
            
            # Adding target column to transformed dataframe
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir    
            
            os.makedirs(transformed_train_dir,exist_ok=True)
            os.makedirs(transformed_test_dir,exist_ok=True)

            # Saving Necessary Train and TEst data 
            transformed_train_data_file_path = os.path.join(transformed_train_dir,"Train.csv")
            transformed_test_data_file_path = os.path.join(transformed_test_dir,"Test.csv")

            ## Saving transformed train and test file
            logging.info("Saving Transformed Train and Transformed test Data")
            
            save_data(file_path = transformed_train_data_file_path, data = train_data)
            save_data(file_path = transformed_test_data_file_path, data = test_data)
            logging.info("Train and Test Data  saved")
           
           
                         ###############################################################
            
            ### Saving FFeature engineering and preprocessor object 
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)


            data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path = transformed_train_data_file_path,
            transformed_test_file_path = transformed_test_data_file_path,
            feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")