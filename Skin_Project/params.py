import os

IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE'))
CHEMIN_1 = os.environ.get('CHEMIN_1')
CHEMIN_2 = os.environ.get('CHEMIN_2')
CHEMIN_3 = os.environ.get('CHEMIN_3')
CHEMIN_4 = os.environ.get('CHEMIN_4')
CHEMIN_METADATA = os.environ.get('CHEMIN_METADATA')
MODEL_TARGET = os.environ.get("MODEL_TARGET")
THRESHOLD = os.environ.get('THRESHOLD')
CLASSIFICATION = str(os.environ.get("CLASSIFICATION"))
SAMPLE_SIZE = float(os.environ.get('SAMPLE_SIZE',0.5))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
