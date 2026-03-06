import numpy as np
import onnxruntime as ort
import cv2
from config import LABEL_MAPPING, NORM_MEAN, NORM_STD

class PiecePredictor:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.label_mapping = LABEL_MAPPING
        mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)
        self._mean = mean
        self._std = std

    def predict(self, squares):
        """
        Predicts the piece on each square.
        Args:
            squares: A numpy array of shape (64, H, W, 3) in BGR uint8.
        Returns:
            A list of 64 string labels for the pieces.
        """
        # Convert BGR -> RGB, normalize to [0,1], apply mean/std, transpose to NCHW
        rgb = squares[:, :, :, ::-1].astype(np.float32) / 255.0
        nchw = np.transpose(rgb, (0, 3, 1, 2))
        nchw = (nchw - self._mean) / self._std

        predictions = self.session.run([self.output_name], {self.input_name: nchw})[0]
        return [self.label_mapping[int(np.argmax(p))] for p in predictions]
