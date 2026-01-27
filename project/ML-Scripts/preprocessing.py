import os
import argparse
import boto3
from botocore.client import Config
from io import BytesIO
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_data_from_s3(bucket_name, file_name):
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    s3 = boto3.resource(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    bucket = s3.Bucket(bucket_name)
    
    obj = bucket.Object(file_name).get()
    df = pd.read_csv(BytesIO(obj["Body"].read()))
    
    return df


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="preprocessing") as preproc_run:

        run_id = preproc_run.info.run_id
        with open("/tmp/run_id.txt", "w") as f:
            f.write(run_id)

        df = load_data_from_s3(args.bucket_name, args.filename)

        # Rename columns
        new_columns_names = ["row_id", "order_id", "ship_mode", "customer_id", "customer_name", "segment", "country", "city", "state", "postal_code", "region", "product_id", 
            "category", "sub_category", "product_name", "sales", "quantity", "discount", "profit", "order_year", "order_month", "order_day", "ship_year", "ship_month", "ship_day"]

        print(f"Loaded columns: {len(df.columns)}, \n Expexted: {len(new_columns_names)}")

        if len(df.columns) == len(new_columns_names):
            df.columns = new_columns_names
        
        df = df.dropna()

        columns_to_str = ["order_id", "ship_mode", "customer_id", "customer_name", "segment", "country", "region",
            "product_id", "category", "sub_category", "product_name"]

        valid_cols_to_str = []
        for column in columns_to_str:
            if column in df.columns: valid_cols_to_str.append(column)

        for col in valid_cols_to_str:
            df[col] = df[col].astype(str)

        # Feature Engineering
        df["quantity"] = df["quantity"].astype(int)

        df["order_date"] = pd.to_datetime(
            df["order_year"].astype(str) + '-' +
            df["order_month"].astype(str) + '-' +
            df["order_day"].astype(str)             
        )

        df["order_weekday_index"] = df["order_date"].dt.weekday
        df["order_weekday"] = df["order_date"].dt.day_name()
        df["order_week_of_year"] = df["order_date"].dt.isocalendar().week
        df["order_is_weekend"] = df["order_weekday_index"] >= 5

        df["ship_date"] = pd.to_datetime(
            df["ship_year"].astype(str) + '-' +
            df["ship_month"].astype(str) + '-' +
            df["ship_day"].astype(str)             
        )

        df["original_price_per_unit"] = df["sales"] / (df["quantity"] * (1- df["discount"]))
        df["markdown_amount"] = (df["original_price_per_unit"] * df["quantity"]) - df["sales"]

        # Columns for prediction
        target_col = "profit"

        categorical_cols = [
            "sub_category",
            "region",
            "segment",
            "order_month"
        ]

        numeric_cols = [
            "sales",
            "quantity",
            "original_price_per_unit",
            "markdown_amount"
        ]

        X = df[categorical_cols + numeric_cols]
        y = df[target_col]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        os.makedirs(args.output_path, exist_ok=True)

        pd.DataFrame(X_train_proc.toarray() if hasattr(X_train_proc, "toarray") else X_train_proc)\
            .assign(target=y_train.values)\
            .to_csv(os.path.join(args.output_path, "train.csv"), index=False)

        pd.DataFrame(X_val_proc.toarray() if hasattr(X_val_proc, "toarray") else X_val_proc)\
            .assign(target=y_val.values)\
            .to_csv(os.path.join(args.output_path, "val.csv"), index=False)

        mlflow.log_params({
            "test_size": args.test_size,
            "random_state": args.random_state,
            "num_categorical_features": len(categorical_cols),
            "num_numeric_features": len(numeric_cols),
        })

        mlflow.log_metrics({
            "train_rows": len(X_train),
            "val_rows": len(X_val),
        })

        mlflow.log_artifacts(args.output_path, artifact_path="processed_data")

        mlflow.sklearn.log_model(
            preprocessor,
            artifact_path="preprocessing",
        )


if __name__ == "__main__":
    main()
