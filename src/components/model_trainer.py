import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('inside Model trainer class initiate_model_trainer method')
            logging.info('split training and test input data')
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Xgboost":XGBRegressor(),
                "Linear Regression":LinearRegression(),
                "Cat boost":CatBoostRegressor(verbose=False),
                "Ada boost":AdaBoostRegressor(),
                "k-Neighbour Regressor":KNeighborsRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
            }
            model_report:dict=evaluate_models(x1=x_train,y1=y_train,models=models,x2=x_test,y2=y_test)

            best_model_score= max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("found the best model")
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return r2_score(y_true=y_test,y_pred=best_model.predict(x_test))
        except Exception as e:
            raise CustomException(e,sys)