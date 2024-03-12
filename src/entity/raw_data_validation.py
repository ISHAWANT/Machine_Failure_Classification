from src.exceptions import ApplicationException
from src.logger import logging
import os, sys
from src.utils.utils import read_yaml_file
import pandas as pd
import collections
import yaml


class IngestedDataValidation:

    def __init__(self, validate_path, schema_path):
        try:
            self.validate_path = validate_path
            self.schema_path = schema_path
            self.data = read_yaml_file(self.schema_path)
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def validate_filename(self, file_name)->bool:
        try:
            print(self.data["FileName"])
            schema_file_name = self.data['FileName']
            if schema_file_name == file_name:
                return True
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def missing_values_whole_column(self)->bool:
        try:
            df = pd.read_csv(self.validate_path)
            count = 0
            for columns in df:
                if (len(df[columns]) - df[columns].count()) == len(df[columns]):
                    count+=1
            return True if (count == 0) else False
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def replace_null_values_with_nan(self)->bool:
        try:
            df = pd.read_csv(self.validate_path)
            df.fillna('NULL',inplace=True)
        except Exception as e:
            raise ApplicationException(e,sys) from e

    
    def check_column_names(self)->bool:
        try:
            df = pd.read_csv(self.validate_path)
            df_column_names = df.columns
  
            schema_column_names = list(self.data['ColumnNames'].keys())
           

            return True if (collections.Counter(df_column_names) == collections.Counter(schema_column_names)) else False

        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise ApplicationException(e,sys)
        
    def validate_data_types(self,filepath):
        flag = True  # Initialize the flag as True
        df = pd.read_csv(filepath)
        
        schema_path=self.schema_path
        
        # Read the schema from YAML file
        with open(schema_path, "r") as file:
            schema_data = yaml.safe_load(file)


        column_names = schema_data["ColumnNames"]
        
        print(column_names)

        for column, expected_type in column_names.items():
            if column not in df.columns:
                print(f"Column '{column}' not found in the dataset.")
            if not df[column].dtype == expected_type:
                flag = False  # Set flag to False if there is a data type mismatch
                print(f"Data type mismatch for column '{column}'. Expected {expected_type}, but found {df[column].dtype}.")

        return flag