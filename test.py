import sys
from src.crop_recommendation.logger import logging
from src.crop_recommendation.exception import CustomException
from src.crop_recommendation.components.data_ingestion import DataIngestion
from src.crop_recommendation.components.data_ingestion import DataIngestionConfig

if __name__=="__main__":

    logging.info("The execution has started")

    try:
        #Data Ingestion
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)