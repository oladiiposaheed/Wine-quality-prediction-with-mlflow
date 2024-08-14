import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging
from urllib.parse import urlparse
import os
import sys
import warnings

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#Evaluate the model
def eval_metrics(actual, pred):
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred)
    return r2, mae, rmse

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(41)

    #Read the wine quality dataset
    csv_data = 'winequality-red.csv'
    
    try:
        data = pd.read_csv(csv_data)
    except Exception as e:
        logger.exception('No dataset is available in your repository, try again')

    #Split the dataset into training and test sets
    train, test = train_test_split(data)

    #
    X_train = train.drop('quality', axis=1)
    X_test = test.drop('quality', axis=1)
    y_train = train['quality']
    y_test = test['quality']
    X = data.drop('quality', axis=1)
    y = data['quality']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    # TARGET = ['quality']
    # X = df.drop(columns=TARGET, axis=1)
    # y = df[TARGET]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=41)
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_test)
        (r2, mae, rmse) = eval_metrics(y_test, y_pred)

        print('ElasticNet model (alpha={:.2f}), l1_ratio={:.2f}):'.format(alpha, l1_ratio))
        print(' R2: {}'.format(r2))
        print(' MAE: {}'.format(mae))
        print(' RMSE: {}'.format(rmse))

        # 
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('rmse', rmse)

        predictions = lr.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        #Model registry does not work with file store
        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(
                lr, 'model', registered_model_name='ElasticNetWindModel', signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, 'model', signature=signature)