import os
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    parser.add_argument("--max-relative-error", type=float, default=4)
    parser.add_argument("--p95-relative-error", type=float, default=0.3)
    parser.add_argument("--mae-drift-factor", type=float, default=1.2)
    parser.add_argument("--max-mean-residual", type=float, default=5.0)
    return parser.parse_args()


def load_best_model_and_data(training_run_id, preprocessing_run_id):
    client = MlflowClient()

    training_run = client.get_run(training_run_id)
    experiment_id = training_run.info.experiment_id
    
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{training_run_id}'",
    )

    best_run = min(
        child_runs,
        key=lambda r: r.data.metrics.get("rmse", float("inf")),
    )

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    tmp_dir = tempfile.mkdtemp()
    client.download_artifacts(
        run_id=preprocessing_run_id,
        path="processed_data",
        dst_path=tmp_dir,
    )

    val_path = os.path.join(tmp_dir, "processed_data", "val.csv")
    val_df = pd.read_csv(val_path)

    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"].values

    return best_run, model, X_val, y_val


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    mlflow.set_experiment(args.experiment_name_mlflow)

    with mlflow.start_run(run_name="evaluation") as eval_run:

        run_id = eval_run.info.run_id
        with open("/tmp/run_id.txt", "w") as f:
            f.write(run_id)
    
        client = MlflowClient()
    
        best_run, model, X_val, y_val = load_best_model_and_data(
            args.training_run_id,
            args.preprocessing_run_id
        )

        mlflow.log_param("best_model_run_id", best_run.info.run_id)
        mlflow.log_param("best_model_name", best_run.info.run_name)
        mlflow.log_param("preprocessing_run_id", args.preprocessing_run_id)


        y_pred = model.predict(X_val)
    
        abs_errors = np.abs(y_pred - y_val)
        relative_errors = abs_errors / np.maximum(y_val, 1.0)
        residuals = y_pred - y_val
    
        # Gate 1: max relative error
        gate_max_rel_error = relative_errors.max() <= args.max_relative_error
    
        # Gate 2: p95 relative error
        gate_p95_rel_error = (
            np.percentile(relative_errors, 95) <= args.p95_relative_error
        )
    
        # Gate 3: MAE drift
        val_mae = mean_absolute_error(y_val, y_pred)
        train_mae = best_run.data.metrics.get("mae", val_mae)
        gate_mae_drift = val_mae <= train_mae * args.mae_drift_factor
    
        # Gate 4: residual bias
        gate_residual_bias = abs(residuals.mean()) <= args.max_mean_residual
    
        # Gate 5: score range sanity
        gate_score_range = (
            (y_pred >= 0).all() and (y_pred <= 100).all()
        )
    
        promote = all([
            gate_max_rel_error,
            gate_p95_rel_error,
            gate_mae_drift,
            gate_residual_bias,
            gate_score_range,
        ])
    
        # ---- Decision artifact
        decision = {
            "best_model_run_id": best_run.info.run_id,
            "best_model_name": best_run.info.run_name,
            "metrics": {
                "val_mae": float(val_mae),
                "train_mae": float(train_mae),
                "max_relative_error": float(relative_errors.max()),
                "p95_relative_error": float(np.percentile(relative_errors, 95)),
                "mean_residual": float(residuals.mean()),
            },
            "gates": {
                "max_relative_error_passed": bool(gate_max_rel_error),
                "p95_relative_error_passed": bool(gate_p95_rel_error),
                "mae_drift_passed": bool(gate_mae_drift),
                "residual_bias_passed": bool(gate_residual_bias),
                "score_range_passed": bool(gate_score_range),
            },
            "promote": promote,
        }
    
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("max_relative_error", float(relative_errors.max()))
        mlflow.log_metric("p95_relative_error", float(np.percentile(relative_errors, 95)))
        mlflow.log_metric("mean_residual", float(residuals.mean()))

        mlflow.log_param("gate_max_relative_error_passed", gate_max_rel_error)
        mlflow.log_param("gate_p95_relative_error_passed", gate_p95_rel_error)
        mlflow.log_param("gate_mae_drift_passed", gate_mae_drift)
        mlflow.log_param("gate_residual_bias_passed", gate_residual_bias)
        mlflow.log_param("gate_score_range_passed", gate_score_range)
        mlflow.log_param("promote", promote)
        
        os.makedirs("evaluation", exist_ok=True)
        with open("evaluation/decision.json", "w") as f:
            json.dump(decision, f, indent=2)
    
        mlflow.log_artifact("evaluation/decision.json", artifact_path="evaluation")

        if not promote:
            raise RuntimeError(
                "Model rejected by evaluation gates. See evaluation/decision.json"
            )


if __name__ == "__main__":
    main()
