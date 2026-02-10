from src.config import Config
import pandas as pd
import logging
import numpy as np  


print("DEBUG CONFIG FILE:", Config.__module__)
print("DEBUG DATA PATH:", getattr(Config, "DATA_PATH", "NO DATA_PATH"))
print("DEBUG FILEPATH:", getattr(Config, "filepath", "NO FILEPATH"))

def data_ingestion():
    df = pd.read_csv(Config.DATA_PATH)
    return df