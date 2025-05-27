import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run, set_tags

mlflow.set_tracking_uri("http://localhost:5000")  # importante
mlflow.set_experiment("mi_tercer_experimento") 

if __name__ == '__main__':
    with start_run():
        # Registrar tags del experimento
        set_tags({
            "author": "Luis",
            "stage": "development",
            "version": "v1.0"
        })

        # Registrar parámetros, métricas y artefactos
        log_param("threshold", 3)
        log_metric("timestamp", 1000)
        log_metric("RAM", 8)
        log_artifact("produced-dataset.csv")
