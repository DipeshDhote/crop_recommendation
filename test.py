import sys
from src.crop_recommendation.logger import logging
from src.crop_recommendation.exception import CustomException
from src.crop_recommendation.components.data_ingestion import DataIngestion
from src.crop_recommendation.components.data_ingestion import DataIngestionConfig
from src.crop_recommendation.components.data_transformation import DataTransformation
from src.crop_recommendation.components.data_transformation import DataTransformationConfig
from src.crop_recommendation.components.model_tranier import ModelTrainer
from src.crop_recommendation.components.model_tranier import ModelTrainerConfig



if __name__=="__main__":

    logging.info("The execution has started")

    try:
        # Data Ingestion
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        # Data Transformation 
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
        # Model Trainer
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)