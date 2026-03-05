import cv2
import numpy as np
import onnxruntime as ort


class YoloBoardDetector:
    def __init__(self, model_path, conf_threshold=0.4, input_size=640):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.input_size = input_size

    def detect(self, image):
        """
        Detect a chessboard in the image.
        Args:
            image: BGR numpy array (OpenCV format).
        Returns:
            (top_left, bottom_right) tuple of (x, y) ints, or None.
        """
        orig_h, orig_w = image.shape[:2]
        tensor, ratio, pad_w, pad_h = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})

        # YOLOv8 ONNX output shape: (1, 5, num_detections)
        # Rows: x_center, y_center, w, h, conf (single-class)
        preds = outputs[0][0]  # (5, N)
        if preds.shape[0] == 5:
            preds = preds.T  # (N, 5)

        if len(preds) == 0:
            return None

        # Filter by confidence (class score is column 4 for single-class)
        scores = preds[:, 4]
        mask = scores >= self.conf_threshold
        preds = preds[mask]
        scores = scores[mask]

        if len(preds) == 0:
            return None

        # Convert xywh to xyxy
        boxes = np.empty_like(preds[:, :4])
        boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2
        boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2
        boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2
        boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2

        # NMS
        keep = self._nms(boxes, scores, iou_threshold=0.5)
        if len(keep) == 0:
            return None

        best = boxes[keep[0]]

        # Undo letterbox padding and scaling
        x1 = int((best[0] - pad_w) / ratio)
        y1 = int((best[1] - pad_h) / ratio)
        x2 = int((best[2] - pad_w) / ratio)
        y2 = int((best[3] - pad_h) / ratio)

        # Clamp to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1), (x2, y2)

    def _preprocess(self, image):
        """Letterbox resize + normalize to NCHW float32 tensor."""
        h, w = image.shape[:2]
        s = self.input_size
        ratio = min(s / h, s / w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_w = (s - new_w) // 2
        pad_h = (s - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_h, s - new_h - pad_h, pad_w, s - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1]
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], ratio, pad_w, pad_h

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.5):
        """Simple greedy NMS. Returns indices sorted by score."""
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            order = rest[iou < iou_threshold]
        return keep
