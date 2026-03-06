import numpy as np
import onnxruntime as ort
from config import LABEL_MAPPING, NORM_MEAN, NORM_STD

_INFERENCE_BATCH = 8  # Run inference in small batches to cap activation memory


def _rss_mb():
    """Current process RSS in MB (Linux /proc; returns -1 if unavailable)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1


class PiecePredictor:
    def __init__(self, model_path):
        print(f"[MEM] before loading piece model: {_rss_mb():.1f} MB", flush=True)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.label_mapping = LABEL_MAPPING
        self._mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)
        print(f"[MEM] after loading piece model: {_rss_mb():.1f} MB", flush=True)

    def predict(self, squares):
        """
        Predicts the piece on each square.
        Args:
            squares: A numpy array of shape (64, H, W, 3) in BGR uint8.
        Returns:
            A list of 64 string labels for the pieces.
        """
        n, h, w = squares.shape[0], squares.shape[1], squares.shape[2]
        print(f"[MEM] predict start (n={n}, {h}x{w}): {_rss_mb():.1f} MB", flush=True)

        # Pre-allocate NCHW float32 directly, copying channels in BGR->RGB order.
        batch = np.empty((n, 3, h, w), dtype=np.float32)
        batch[:, 0] = squares[:, :, :, 2]  # R
        batch[:, 1] = squares[:, :, :, 1]  # G
        batch[:, 2] = squares[:, :, :, 0]  # B
        batch /= 255.0
        batch -= self._mean
        batch /= self._std
        print(f"[MEM] after preprocessing batch ({batch.nbytes / 1e6:.1f} MB array): {_rss_mb():.1f} MB", flush=True)

        # Run inference in small batches so ORT activation memory stays bounded.
        outputs = []
        for i in range(0, n, _INFERENCE_BATCH):
            chunk = batch[i:i + _INFERENCE_BATCH]
            outputs.append(self.session.run([self.output_name], {self.input_name: chunk})[0])
            print(f"[MEM]   batch {i//_INFERENCE_BATCH+1}/{-(-n//_INFERENCE_BATCH)}: {_rss_mb():.1f} MB", flush=True)

        all_preds = np.concatenate(outputs, axis=0)
        print(f"[MEM] predict done: {_rss_mb():.1f} MB", flush=True)
        return [self.label_mapping[int(np.argmax(p))] for p in all_preds]
