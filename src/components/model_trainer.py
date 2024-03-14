
import logging
import sys
import os
import yaml
import pandas as pd
from src.logger import logging
from src.exceptions import ApplicationException
from src.utils.utils import save_object,load_numpy_array_data
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.artifact_entity import ModelTrainerArtifact
from src.constants import *
import sys
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score


class ModelClassifier:
    def __init__(self):
        self.model_dict = {
            'LightGBM': lgb.LGBMClassifier(verbose=0),
            'CatBoost': CatBoostClassifier(silent=True, iterations=50),
            'XgBoost': xgb.XGBClassifier()
        }
        self.params = {
                        'LightGBM': {
                            'num_leaves': (10, 60),
                            'learning_rate': (0.01, 0.5),
                            'min_data_in_leaf': (10, 50),
                            'feature_fraction': (0.1, 0.9), 
                            'bagging_fraction': (0.5, 1.0)   
                        },
                        'CatBoost': {
                            'depth': (6, 16),
                            'learning_rate': (0.0, 0.5),
                            'l2_leaf_reg': (1, 10),
                            'iterations': (60, 60)
            
                        },
                        'XgBoost': {
                            'max_depth': (3, 50),
                            'learning_rate': (0.01, 0.4),
                            'n_estimators': (100, 500),
                            'min_child_weight': (1, 10)   
                        }
                    }   

    def fit_with_params(self, model_name, X, y):
        model = self.model_dict.get(model_name)
        if model:
            params = self.params.get(model_name, None)
            if params:
                print(f"Fitting {model_name} with specified parameters...")
                model.set_params(**params)
            else:
                print(f"Fitting {model_name} with default parameters.")

            return model.fit(X, y)
        else:
            raise ValueError(f"Model {model_name} not found. Please set the model first.")

    def fit_without_params(self, model_name, X, y):
        model = self.model_dict.get(model_name)
        if model:
            print(f"Fitting {model_name} without parameters.")
            return model.fit(X, y)
        else:
            raise ValueError(f"Model {model_name} not found. Please set the model first.")

class OptunaTuner_Catboost:
    def __init__(self, model, params, X_train, X_test, y_train, y_test):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def Objective(self, trial):
        param_values = {}
        for key, value_range in self.params.items():
            if isinstance(value_range, tuple):  # Check if value_range is a tuple
                if value_range[0] <= value_range[1]:
                    if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                        param_values[key] = trial.suggest_int(key, value_range[0], value_range[1])
                    else:
                        param_values[key] = trial.suggest_float(key, value_range[0], value_range[1])
                else:
                    raise ValueError(f"Invalid range for {key}: low={value_range[0]}, high={value_range[1]}")
            else:  # If value_range is not a tuple, treat it as a single value
                param_values[key] = value_range

        model = CatBoostClassifier(**param_values)
        
        # Train the model on the training data
        model.fit(self.X_train, self.y_train)

        # Evaluate the model using AUC-ROC on the test data
        y_probs = model.predict_proba(self.X_test)[:, 1]  # Get predicted probabilities for the positive class
        auc_roc = roc_auc_score(self.y_test, y_probs)

        return auc_roc

    def tune(self, n_trials=100):
        study = optuna.create_study(direction="maximize")  # maximize AUC-ROC

        # Perform Optuna tuning
        study.optimize(self.Objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        # Create a new CatBoost model instance with the best parameters
        best_model = CatBoostClassifier(**best_params)

        # Train the best model on the whole training dataset
        best_model.fit(self.X_train, self.y_train)

        # Evaluate the best model using AUC-ROC on the test dataset
        y_probs = best_model.predict_proba(self.X_test)[:, 1]  # Get predicted probabilities for the positive class
        best_auc_score = roc_auc_score(self.y_test, y_probs)
        print(f"Best AUC Score: {best_auc_score}")

        # Here, we return both the tuned model and the best AUC-ROC score
        return best_auc_score, best_model,best_params

class OptunaTuner:
    def __init__(self, model, params, X_train, y_train, X_test, y_test):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def Objective(self, trial):
        param_values = {}
        for key, value_range in self.params.items():
            if value_range[0] <= value_range[1]:
                if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                    param_values[key] = trial.suggest_int(key, value_range[0], value_range[1])
                else:
                    param_values[key] = trial.suggest_float(key, value_range[0], value_range[1])
            else:
                raise ValueError(f"Invalid range for {key}: low={value_range[0]}, high={value_range[1]}")

        self.model.set_params(**param_values)

        # Fit the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # Evaluate the model using AUC-ROC on the test data
        y_probs = self.model.predict_proba(self.X_test)[:, 1]
        auc_roc = roc_auc_score(self.y_test, y_probs)

        return auc_roc

    def tune(self, n_trials=100):
        study = optuna.create_study(direction="maximize")  # maximize AUC-ROC score
        study.optimize(self.Objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        # Set the best parameters to the model
        self.model.set_params(**best_params)

        # Retrain the model with the best parameters on the entire training set
        self.model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test set using AUC-ROC
        y_probs_test = self.model.predict_proba(self.X_test)[:, 1]
        best_auc_score = roc_auc_score(self.y_test, y_probs_test)
        print(f"Best AUC Score on Test Set: {best_auc_score}")

        # Here, we return the tuned model and the best AUC-ROC score on the test set
        return best_auc_score, self.model,best_params
class VotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
        self.fitted_models = []

    def fit(self, X, y):
        self.fitted_models = []
        for _, model in self.estimators:
            model.fit(X, y)
            self.fitted_models.append(model)
        return self

    def predict(self, X):
        if self.voting == 'hard':
            predictions = np.asarray([model.predict(X) for _, model in self.estimators])
            return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        elif self.voting == 'soft':
            probabilities = np.asarray([model.predict_proba(X) for _, model in self.estimators])
            return np.argmax(np.mean(probabilities, axis=0), axis=1)
        else:
            raise ValueError(f"Invalid voting method: {self.voting}. Use 'hard' or 'soft'.")

    def get_fitted_models(self):
        return self.fitted_models

class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainingConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
   
        except Exception as e:
            raise ApplicationException(e, sys) from e

        
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding transformed Training and Test")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
        
            train_df = pd.read_csv(transformed_train_file_path)
            test_df = pd.read_csv(transformed_test_file_path)
            
            logging.info(" Train and Test dtaframe loaded")
                        
            # Target 
            Target_Column = 'Machine_failure'
            
            X_train=train_df.drop(columns=Target_Column,axis=1)
            X_test=test_df.drop(columns=Target_Column,axis=1)
            y_train=train_df[Target_Column]
            y_test=test_df[Target_Column]
            
            
            models = ModelClassifier()

            results = {}  # Dictionary to store the best models and their AUC scores

            # List to store the tuned models
            tuned_models = []

            for model_name, model in models.model_dict.items():
                if model_name == 'CatBoost':

                    logging.info(f"Tuning and fitting model ----------->>>>  {model_name}")

                    # Create an instance of OptunaTuner for each model
                    tuner = OptunaTuner_Catboost(model, params=models.params[model_name], X_train=X_train,y_train=y_train,
                                                                                            X_test=X_test,y_test=y_test)

                    # Perform hyperparameter tuning
                    best_auc_score, tuned_model,best_params = tuner.tune(n_trials=10)

                    logging.info(f"Best AUC score for {model_name}: {best_auc_score}")
                    logging.info("----------------------")

                    # Append the tuned model to the list of tuned models
                    tuned_models.append((model_name, tuned_model))
                    
                    results[model_name] = best_auc_score
                    
                    
                else: 
                    
                    logging.info(f"Tuning and fitting model ----------->>>>  {model_name}")

                    # Create an instance of OptunaTuner for each model
                    tuner = OptunaTuner(model, params=models.params[model_name], X_train=X_train,y_train=y_train,
                                                                                X_test=X_test,y_test=y_test)
                    # Perform hyperparameter tuning
                    best_auc_score, tuned_model,best_params = tuner.tune(n_trials=10)

                    logging.info(f"Best AUC score for {model_name}: {best_auc_score}")
                    logging.info(f"Best Params {model_name}: {best_params}")
                    
                    logging.info("----------------------")

                    # Append the tuned model to the list of tuned models
                    tuned_models.append((model_name, tuned_model))
                    
                    results[model_name] = best_auc_score
                    
            # Convert the 'results' dictionary to a DataFrame
            result_df = pd.DataFrame(results.items(), columns=['Model', 'Best AUC Score'])
            
            
            logging.info(f" Prediction Done : {result_df}")
            
                        
            logging.info(f"-------------")
            
            ensemble=VotingEnsemble(estimators=tuned_models,voting='soft')
            ensemble.fit(X_train,y_train)
            
            # Make predictions on X_test using the ensemble
            y_pred_probs = ensemble.predict(X_test)  # Probability of positive class (class 1)

            # Calculate the AUC (Area Under the ROC Curve)
            auc = roc_auc_score(y_test, y_pred_probs)
            
            logging.info(f" Auc Score of Test data : {auc}")


            trained_model_file_path = self.model_trainer_config.model_object_file_path
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=ensemble)
          
            model_report = {'Model_name': 'Voting_Ensemble', 'Auc_Score': str(auc)}
            
            # Model Report 
            report=model_report
            logging.info(f"Dumping Score in report.....")
            # Save report in artifact folder
            model_artifact_report_path = self.model_trainer_config.model_report
            with open(model_artifact_report_path, 'w') as file:
                yaml.safe_dump(report, file)
            logging.info("-----------------------")
            
            model_trainer_artifact = ModelTrainerArtifact(
      
                                                trained_model_file_path=trained_model_file_path,
                                                model_artifact_report=model_artifact_report_path,
                                         
                                            )
        
            

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
            
            