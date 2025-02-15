import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class VehiclePricePredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.preprocessor = None
        self.feature_columns = ["marka", "seri", "model", "yil", "km"]
        self.target_column = "fiyat"

    def prepare_data(self):
        df = pd.read_csv(self.csv_path)

        # Clean and convert numeric fields
        df["yil"] = df["yil"].astype(int)
        df["km"] = df["km"].astype(float)
        df["fiyat"] = df["fiyat"].astype(float)

        # Filter realistic values
        df = df[(df["fiyat"] > 10000) & (df["fiyat"] < 5000000)]
        df = df[(df["km"] > 0) & (df["km"] < 500000)]
        df = df[df["yil"] > 1990]

        return df

    def build_pipeline(self):
        categorical_features = ["marka", "seri", "model"]
        numeric_features = ["yil", "km"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=7,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )

        return pipeline

    def train(self, test_size=0.2):
        df = self.prepare_data()
        X = df[self.feature_columns]
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        # Save model
        joblib.dump(pipeline, "src/pricing/price_model.pkl")

        return pipeline

    def load_model(self):
        self.model = joblib.load("src/pricing/price_model.pkl")
        return self.model

    def predict(self, brand, model, trim, year, km):
        input_data = pd.DataFrame(
            [{"marka": brand, "seri": model, "model": trim, "yil": year, "km": km}]
        )

        return self.model.predict(input_data)[0]
