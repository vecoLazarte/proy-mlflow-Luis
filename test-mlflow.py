import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run

mlflow.set_tracking_uri("http://localhost:5000")  # importante

if __name__ == '__main__':
    with start_run():
        log_param("threshold", 3)
    
        log_metric("timestamp", 1000)

        log_artifact("produced-dataset.csv")