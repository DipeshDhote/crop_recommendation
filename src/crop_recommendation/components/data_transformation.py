import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.crop_recommendation.logger import logging
from src.crop_recommendation.exception import CustomException
from src.crop_recommendation.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    

            # Define the custom ranking for each ordinal variable
            

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("reading complete numerical pipeline")
            
            logging.info("Numerical pipeline completed")
            

            preprocessor = ColumnTransformer(
                [
                    
                    ("numerical_pipeline",num_pipeline,numerical_columns)
                    
                ]
            )

            logging.info("Column transformation completed")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            

            logging.info(f"Training data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="label"
            drop_columns = [target_column_name]

            input_features_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

        

            logging.info("Applying preprocessing on training and test data")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Preprocessing completed")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object to {self.data_transformation_config.preprocessor_obj_file_path}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
