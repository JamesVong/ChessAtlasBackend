import torch
from config import LABEL_MAPPING

# Normalization stats computed from train split (see chess_atlas_v1 training notebook)
_NORM_MEAN = torch.tensor([0.7312, 0.6766, 0.5781], dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor([0.2291, 0.2138, 0.2072], dtype=torch.float32).view(1, 3, 1, 1)


class PiecePredictor:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.label_mapping = LABEL_MAPPING

    def predict(self, squares):
        """
        Predicts the piece on each square.
        Args:
            squares: numpy array of shape (64, H, W, 3), dtype uint8, BGR channel order.
        Returns:
            A list of 64 string labels for the pieces.
        """
        # BGR uint8 HWC -> RGB float32 NCHW, scale to [0, 1]
        rgb = squares[:, :, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).float().div_(255.0)  # (64, H, W, 3)
        tensor = tensor.permute(0, 3, 1, 2)                 # (64, 3, H, W)
        tensor = (tensor - _NORM_MEAN) / _NORM_STD

        with torch.no_grad():
            logits = self.model(tensor)

        indices = logits.argmax(dim=1).numpy()
        return [self.label_mapping[i] for i in indices]
