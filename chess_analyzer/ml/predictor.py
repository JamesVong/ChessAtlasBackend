import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tensorflow as tf
import numpy as np
from config import LABEL_MAPPING

class PiecePredictor:
    def __init__(self, model_path):
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_NUM_INTRAOP_THREADS", "1")))
            tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_NUM_INTEROP_THREADS", "1")))
        except RuntimeError:
            # Threading settings can only be applied before TF runtime initializes.
            pass

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.label_mapping = LABEL_MAPPING

    def predict(self, squares):
        """
        Predicts the piece on each square.
        Args:
            squares: A list of 64 preprocessed square images.
        Returns:
            A list of 64 string labels for the pieces.
        """
        images_np = np.asarray(squares, dtype=np.float32) / 255.0
        predictions = self.model(images_np, training=False).numpy()
        board_labels = [self.label_mapping[str(np.argmax(p))] for p in predictions]
        return board_labels
