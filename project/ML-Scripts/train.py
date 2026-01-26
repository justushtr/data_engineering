import os
import argparse
import tempfile
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    return parser.parse_args()


def load_preprocessed_data(run_id: str):
    """
    Downloads train/val data from MLflow artifacts.
    """
    client = mlflow.tracking.MlflowClient()

    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(
        run_id=run_id,
        path="processed_data",
        dst_path=tmp_dir,
    )

    train_path = os.path.join(tmp_dir, "processed_data", "train.csv")
    val_path = os.path.join(tmp_dir, "processed_data", "val.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    return X_train, X_val, y_train, y_val


def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)
    return {
        "rmse": root_mean_squared_error(y_val, preds),
        "mae": mean_absolute_error(y_val, preds),
    }


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="training") as parent_run:

        run_id = parent_run.info.run_id
        with open("/tmp/run_id.txt", "w") as f:
            f.write(run_id)

        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("preprocessing_run_id", args.preprocessing_run_id)
    
        mlflow.set_tag("data_source", "preprocessed_artifact")
    
        X_train, X_val, y_train, y_val = load_preprocessed_data(
            args.preprocessing_run_id
        )

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                max_depth=20,
                min_samples_leaf = 2,
                min_smaples_split = 5,
                n_estimators =500,
                random_state = 42,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.08,
                random_state=42,
            ),
        }

        best_rmse = float("inf")
        best_model_name = None

        for model_name, model in models.items():

            with mlflow.start_run(
                run_name=model_name,
                nested=True,
            ) as child_run:

                model.fit(X_train, y_train)

                metrics = evaluate(model, X_val, y_val)

                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=None,
                )

                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_model_name = model_name

        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_param("best_model", best_model_name)


if __name__ == "__main__":
    main()
