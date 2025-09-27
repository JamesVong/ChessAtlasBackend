import tensorflow as tf
import numpy as np
from config import LABEL_MAPPING

class PiecePredictor:
    def __init__(self, model_path):
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
        images_np = np.array(squares, dtype='float32') / 255.0
        predictions = self.model.predict(images_np)
        board_labels = [self.label_mapping[str(np.argmax(p))] for p in predictions]
        return board_labels
