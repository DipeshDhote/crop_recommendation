import os
import sys 
import pandas as pd
import numpy as np
from src.crop_recommendation.logger import logging
from src.crop_recommendation.exception import CustomException
from src.crop_recommendation.utils import load_object 



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):       
          try:

            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")            
            print("Before Loading")

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            logging.info("input data scaled and model predict completed")
            
            return preds
            
          except Exception as e:
                raise CustomException(e,sys)




class CustomData:
    def __init__( self,
                 
      N:float,
      P:int,
      K:int,
      temperature:float, 
      humidity:float,
      ph:float,
      rainfall:float,
      ):
            
      self.N= N
      self.P= P
      self.K= K
      self.temperature= temperature
      self.humidity= humidity
      self.ph= ph
      self.rainfall= rainfall
      

    def get_data_as_data_frame(self):
          try:
                custom_data_input_dict = {                                          
                      'N':[self.N],
                      'P':[self.P],
                      'K':[self.K],
                      'temperature':[self.temperature], 
                      'humidity':[self.humidity],
                      'ph':[self.ph],
                      'rainfall':[self.rainfall]
                  }
                
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
                
          except Exception as e:
                raise CustomException(e,sys)
                

