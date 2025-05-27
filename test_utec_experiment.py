import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run, set_tags

mlflow.set_tracking_uri("http://localhost:5000")  # Define la URI (dirección) donde MLflow registrará la información.
mlflow.set_experiment("mi_experimento_utec") # Define el nombre del experimento bajo el cual se agruparán todos los runs.

if __name__ == '__main__':
    with start_run(): # abre un contexto para registrar todo lo que ocurra dentro como un run.
        # Registrar parámetros, métricas y artefactos
        log_param("threshold", 3) # Guarda un valor que se usó en el experimento. Aquí, el parámetro threshold = 3.
        for ram in range(4, 33): 
            log_metric("RAM", ram)
        log_artifact("produced-dataset.csv")
        set_tags({
            "author": "Luis",
            "stage": "development",
            "version": "v1.0"
        })
