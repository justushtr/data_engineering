import os
import json
import argparse
import pandas as pd
import mlflow
import boto3
from botocore.client import Config
from io import BytesIO

EXPECTED_COLUMNS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "math score",
    "reading score",
    "writing score",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--filename", required=True)
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

    EXPECTED_COLUMNS = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
        "math score",
        "reading score",
        "writing score",
    ]

    with mlflow.start_run(run_name="eda") as eda_run:

        run_id = eda_run.info.run_id
        with open("/tmp/run_id.txt", "w") as f:
            f.write(run_id)

        df = load_data_from_s3(args.bucket_name, args.filename)

        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        eda_summary = {
            "num_rows": len(df),
            "num_columns": df.shape[1],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
        }

        categorical_cols = [
            "gender",
            "race/ethnicity",
            "parental level of education",
            "lunch",
            "test preparation course",
        ]

        categorical_distributions = {
            col: df[col].value_counts(normalize=True).to_dict()
            for col in categorical_cols
        }

        target_stats = {
            "math_score_mean": float(df["math score"].mean()),
            "math_score_std": float(df["math score"].std()),
            "math_score_min": int(df["math score"].min()),
            "math_score_max": int(df["math score"].max()),
        }

        os.makedirs("eda", exist_ok=True)

        with open("eda/summary.json", "w") as f:
            json.dump(eda_summary, f, indent=2)

        with open("eda/categorical_distributions.json", "w") as f:
            json.dump(categorical_distributions, f, indent=2)

        with open("eda/target_stats.json", "w") as f:
            json.dump(target_stats, f, indent=2)

        mlflow.log_artifacts("eda", artifact_path="eda")

        mlflow.log_metrics({
            "num_rows": len(df),
            "math_score_mean": target_stats["math_score_mean"],
            "math_score_std": target_stats["math_score_std"],
        })


if __name__ == "__main__":
    main()
