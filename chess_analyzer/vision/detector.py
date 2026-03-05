from ultralytics import YOLO


class YoloBoardDetector:
    def __init__(self, model_path, conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image):
        """
        Detect a chessboard in the image.
        Args:
            image: BGR numpy array (OpenCV format).
        Returns:
            (top_left, bottom_right) tuple of (x, y) ints, or None.
        """
        results = self.model.predict(image, verbose=False, conf=self.conf_threshold)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return None

        # Pick the highest-confidence detection
        best_idx = boxes.conf.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].int().tolist()

        return (x1, y1), (x2, y2)
