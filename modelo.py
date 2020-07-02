""" Train regression model and create csv file with predictions

    Requirements:

        >> pip install -r requirements.txt

    Execution:

        >> python modelo.py train_dataset.csv test_dataset.csv

    Help:

        >> python modelo.py --help
"""
from sklearn.compose import (
    ColumnTransformer, TransformedTargetRegressor
)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, StandardScaler
)
import argparse
import numpy as np
import pandas as pd


def read_data(filename):
    """ Read data and drop duplicated lines
    """
    df_all = pd.read_csv(filename)
    df = df_all.drop_duplicates(subset="movie_views").set_index("movie_id")
    return df


def make_model():
    """ Make pipeline considering a preprocessing step and a linear regression
        with Ridge regularization. Numeric features are standardized, while
        categorical features are encoded. There is also a logarithmic
        transformatin for the revenue features and the target.
    """
    revenue_features = [
        "box_office_revenue",
        "movie_theater_revenue",
    ]
    numeric_features = [
        "budget",
        "duration",
        "user_ratings",
        # "trailer_audience",
        "movie_theater_price",
    ]
    categorical_features = [
        "producer",
        "origin_country",
        "director",
        "genre",
        "main_actor",
        "story_author",
        "year_launched",
    ]
    revenue_transformer = Pipeline(steps=[
        ("log1p", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
    ])
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("rev", revenue_transformer, revenue_features),
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    ridge = TransformedTargetRegressor(
        regressor=RidgeCV(),
        func=np.log1p,
        inverse_func=np.expm1
    )
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ridge", ridge)
    ])
    return model


def train_model(filename):
    """ Train model
    """
    df = read_data(filename)
    X = df.drop("movie_views", axis=1)
    y = df.movie_views
    model = make_model()
    model.fit(X, y)
    return model


def predictions(model, infile, outfile):
    """ Generate predictions for the test dataset
    """
    df_test = pd.read_csv(infile).set_index("movie_id")
    y_pred = model.predict(df_test)
    s = pd.Series(y_pred, index=df_test.index, name="movie_views")
    s.to_csv(outfile, float_format="%.0f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Challenge Solution",
        epilog="Author: Jose Cupertino Ruiz Vargas"
    )
    parser.add_argument("train", help="train dataset")
    parser.add_argument("test", help="test dataset")
    args = parser.parse_args()
    print("Training model.")
    model = train_model(args.train)
    print("Generating predictions.")
    predictions(model, args.test, "predictions.csv")
    print("Check results at predictions.csv")
