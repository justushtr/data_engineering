import os
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name-mlflow", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--preprocessing-run-id", required=True)
    parser.add_argument("--max-relative-error", type=float, default=10)
    parser.add_argument("--p95-relative-error", type=float, default=1.0)
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
    
        ## Calculatios
        val_mae = mean_absolute_error(y_val, y_pred)
        val_r2 = r2_score(y_val, y_pred)
        val_medae = median_absolute_error(y_val, y_pred)

        std_profit = np.std(y_val)

        residuals = y_pred - y_val
        abs_errors = np.abs(y_pred - y_val)
        
        relative_errors = abs_errors / np.maximum(np.abs(y_val), 1e-6)        
        train_mae = best_run.data.metrics.get("mae", val_mae)

        ## Gates
        # Business Gates
        gate_r2_score = val_r2 > 0.60
        gate_mae_vs_std = val_mae < (0.5 * std_profit)

        # Technical Gates
        gate_mae_drift = val_mae <= (train_mae * args.mae_drift_factor)
        gate_residual_bias = abs(residuals.mean()) <= args.max_mean_residual
        gate_p95_rel_error = np.percentile(relative_errors, 95) <= args.p95_relative_error
    
        promote = all([
            gate_r2_score,
            gate_mae_vs_std,
            gate_mae_drift,
            gate_residual_bias,
            gate_p95_rel_error,
        ])
    
        # ---- Decision artifact
        decision = {
            "best_model_run_id": best_run.info.run_id,
            "best_model_name": best_run.info.run_name,
            "metrics": {
                "val_mae": float(val_mae),
                "val_r2": float(val_r2),
                "data_std": float(std_profit),
                "max_abs_error": float(abs_errors.max()),
                "median_abs_error": float(np.median(abs_errors)),
                "mean_residual_bias": float(residuals.mean())
            },
            "gates": {
                "r2_greater_than_0.60": bool(gate_r2_score),
                "mae_smaller_than_half_std": bool(gate_mae_vs_std),
                "mae_drift_passed": bool(gate_mae_drift),
                "residual_bias_passed": bool(gate_residual_bias),
                "p95_rel_error_passed": bool(gate_p95_rel_error)
            },
            "promote": promote,
        }
    
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_medae", val_medae)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("max_abs_error", float(abs_errors.max()))
        mlflow.log_metric("mean_residual", float(residuals.mean()))

        mlflow.log_param("gate_r2_greater_than_0.60_passed", gate_r2_score)
        mlflow.log_param("gate_mae_smaller_than_half_std_passed", gate_mae_vs_std)
        mlflow.log_param("gate_mae_drift_passed", gate_mae_drift)
        mlflow.log_param("gate_residual_bias_passed", gate_residual_bias)
        mlflow.log_param("gate_p95_rel_error_passed", gate_p95_rel_error)

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
