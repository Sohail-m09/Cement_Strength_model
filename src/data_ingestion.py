from src.config import Config
import pandas as pd
import logging
import numpy as np  

def data_ingestion():
    df = pd.read_csv(Config.DATA_PATH)
    return df