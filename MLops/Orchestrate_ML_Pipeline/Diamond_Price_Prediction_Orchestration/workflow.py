from sched import scheduler
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

import mlflow
from prefect import task, flow

@task
def load_data(path: str, cleaning_cols: List[str], unwanted_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.rename(columns={'x':'length', 'y':'width', 'z':'depth', 'depth':'depth%'}, inplace = True)
    df[cleaning_cols] = df[cleaning_cols].replace(0, np.NaN)
    df.dropna(inplace = True)
    df.drop(unwanted_cols, axis = 1, inplace = True)
    return df

@task
def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size = test_data_ratio, random_state=4)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr,'X_TEST': X_te, 'Y_TEST': y_te}

@task
# Seperating Numerical Data
def num_col(df: pd.DataFrame) -> pd.DataFrame:
    sep_numerical = df.select_dtypes(include=['int64', 'float64'])
    return sep_numerical

@task
def get_scaler(df: pd.DataFrame) -> pd.DataFrame:
    # Scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler

@task
def rescale_data(df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    # scaling the numerical features
    # Column names are annoyingly lost after scaling
    # (i.e. the dataframe is coverted to a numpy ndarray)
    data_rescaled = pd.DataFrame(scaler.transform(df),
                                columns = df.columns,
                                index = df.index)
    return data_rescaled

@task
# Seperating Categorical Columns
def cat_col(df: pd.DataFrame) -> pd.DataFrame:
    sep_categorical = df.select_dtypes(include=['object'])
    return sep_categorical

@task
def encoding(df: pd.DataFrame) -> pd.DataFrame:
    cut_encoder = {'Ideal':4, 'Premium':3, 'Very Good':2, 'Good':1, 'Fair':0}
    df['cut'] = df['cut'].apply(lambda x: cut_encoder[x])


    color_encoder = {'G':3, 'E':1, 'F':2, 'H':4, 'D':0, 'I':5, 'J':6}
    df['color'] = df['color'].apply(lambda x: color_encoder[x])
    
    clarity_encoder = {'SI1':2, 'VS2':5, 'SI2':3, 'VS1': 4, 'VVS2':7, 'VVS1':6, 'IF':1, 'I1':0}
    df['clarity'] = df['clarity'].apply(lambda x: clarity_encoder[x])

@task
def concat_df(df: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    concated_df = pd.concat([df, df1], axis = 1)
    return concated_df

@task
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series, estimator: Any, parameters: List) -> Any:
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)

    with mlflow.start_run():        
        reg = GridSearchCV(
            estimator=estimator, 
            param_grid=parameters, 
            scoring='neg_mean_absolute_error',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        reg.fit(X_train, y_train)
        
        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return reg







# Workflow
@flow
def main(path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diamond Price Prediction")


    # Define Parameters
    TARGET_COL = 'price'
    UNWANTED_COLS = ['depth%', 'table']
    CLEANING_COLS = ['length', 'width', 'depth']
    LABELLING_COLS = ['cut', 'color', 'clarity']
    TEST_DATA_RATIO = 0.25
    DATA_PATH = path

    # Loading & Cleaning the Data
    dataframe = load_data(path = DATA_PATH, cleaning_cols = CLEANING_COLS, unwanted_cols = UNWANTED_COLS)

    # Identify Target Variable
    target_data = dataframe[TARGET_COL]
    input_data = dataframe.drop([TARGET_COL], axis = 1)

    # Split the Data into Train and Test
    train_test_dict = split_data(input_ = input_data, output_ = target_data, test_data_ratio = TEST_DATA_RATIO)

    # Preprocessing X_train
    Numerical_train_df = num_col(train_test_dict['X_TRAIN'])
    Categorical_train_df = cat_col(train_test_dict['X_TRAIN'])
    X_train_cat_le = encoding(Categorical_train_df)
    concated_df1 = concat_df(Numerical_train_df, X_train_cat_le)
    scaler = get_scaler(concated_df1)
    X_train_transformed = rescale_data(df = concated_df1, scaler = scaler)

    # Preprocessing X_test
    Numerical_test_df = num_col(train_test_dict['X_TEST'])
    Categorical_train_df = cat_col(train_test_dict['X_TEST'])
    X_test_cat_le = encoding(Categorical_train_df)
    concated_df2 = concat_df(Numerical_test_df, X_test_cat_le)
    scaler = get_scaler(concated_df2)
    X_test_transformed = rescale_data(df = concated_df2, scaler = scaler)

    # Model Training
    ESTIMATOR = KNeighborsRegressor()
    HYPERPARAMETERS = [{'n_neighbors':[i for i in range(1, 51)], 'p':[1, 2]}]

    regressor = find_best_model(X_train_transformed, train_test_dict['Y_TRAIN'], ESTIMATOR, HYPERPARAMETERS)
    print(regressor.best_params_)
    print(regressor.score(X_test_transformed, train_test_dict['Y_TEST']))

# Deploy the main function
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

deployment = Deployment.build_from_flow(
    flow = main,
    name = "Diamond_Price_Prediction",
    schedule = IntervalSchedule(interval = timedelta(days=7)),
    work_queue_name = "ml"
)

deployment.apply()


# Run the main function
main(path = r'data\diamonds.csv')