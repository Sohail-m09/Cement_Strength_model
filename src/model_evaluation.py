from src.config import Config

import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
import pickle
import os

import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def model_evaluation(models, X_train, y_train, X_test, y_test):
    best_score = -float('inf')
    best_model = None
    best_model_name = ""

    print(f"{'Model Name':<25} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 50)

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and score
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{name:<25} | {rmse:<10.4f} | {r2:<10.4f}")

        # Logic to pick the best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    print("-" * 50)
    print(f"WINNER: {best_model_name} with R2: {best_score:.4f}")
    
    return best_model, best_model_name