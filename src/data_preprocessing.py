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
def data_preprocessing(df):
        # 1. Prepare Features (X) and Target (y)
    X = df.drop('strength', axis=1)
    y = df['strength']

    # 2. Split the data (train_and_test_split logic)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # 3. Initialize and Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predict
    y_pred = model.predict(X_test)

    # 5. Calculate Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 6. Store result (matching the list format in your function)
    result = ['LinearRegression', rmse, r2]

    # Output the result
    return X_train, X_test, y_train, y_test