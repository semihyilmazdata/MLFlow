import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__":
    with mlflow.start_run(run_name = 'MLflow introduction'):
        data_path = "path/to/dataset.csv"
        model_path = "path/to/save/model"


        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        X, y = data, target
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squared = model.score(X, y)
        mae = mean_absolute_error(y_test, y_pred)

        print(mse)
        print(r_squared)
        print(mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r_squared)
        mlflow.log_metric("mae", mae)

    
        mlflow.sklearn.log_model(model, 'Linear Regression with ML Flow')

