import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
)
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()
    
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
                                        
            models ={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }
                      
            model_report:dict = evaluate_models(X_train = X_train,y_train=y_train,X_test = X_test,y_test=y_test,models=models)
            ##to get the best models
            best_model_score = max(sorted(model_report.values()))
            
            ## to get best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")
            
            save_object(
                file_path = self.trained_model_config.trained_model_file_path,
                obj = best_model)
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
        
