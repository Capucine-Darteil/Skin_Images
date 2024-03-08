import time
import os
import glob

from tensorflow import keras
import mlflow
from Skin_Project.params import *
from mlflow.tracking import MlflowClient

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(CHEMIN_4, f"{timestamp}.h5")
    model.save(model_path)
    print("Model saved locally")

    if MODEL_TARGET == "mlflow":
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name="skin project model"
        )

        print("✅ Model saved to MLflow")

        return None

    return None



def load_best_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{CHEMIN_4}/best_model.h5")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("Model loaded from local disk")

        return latest_model
    pass



def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":

        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{CHEMIN_4}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print('local_model_paths',local_model_paths)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("Model loaded from local disk")

        return latest_model



    elif MODEL_TARGET == "mlflow":

        # load model from ML Flow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(name="skin project model", stages=[stage])
            model_uri = model_versions[0].source

            assert model_uri is not None

        except:
            print(f"\n❌ No model found with name 'skin project model' in stage {stage}")

            return None

        model = mlflow.tensorflow.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        return model
    else:
        return None



    return None
