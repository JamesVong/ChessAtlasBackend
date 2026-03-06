import cv2
import numpy as np
import base64
import time
from config import MAX_IMAGE_DIM, PIECE_TO_FEN
from chess_analyzer.vision import preprocessing


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


class ChessAnalysisService:
    VALID_ORIENTATIONS = {"white": "White", "black": "Black"}

    def __init__(self, detector, predictor):
        self.detector = detector
        self.predictor = predictor

    def _convert_to_fen(self, board_labels, orientation="White"):
        if orientation == "Black":
            board_labels = board_labels[::-1]

        fen_rows = []
        for i in range(0, 64, 8):
            row_pieces = board_labels[i:i+8]
            fen_row = ""
            empty_count = 0
            for piece in row_pieces:
                symbol = PIECE_TO_FEN.get(piece, "1")
                if symbol == "1":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += symbol
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        return "/".join(fen_rows)

    def _normalize_orientation(self, orientation):
        if orientation is None:
            return "White"
        return self.VALID_ORIENTATIONS.get(str(orientation).strip().lower())

    def analyze_image(self, image_bytes, include_cropped_image=True, orientation="White"):
        t0 = time.perf_counter()
        print(f"[MEM] request start: {_rss_mb():.1f} MB", flush=True)

        normalized_orientation = self._normalize_orientation(orientation)
        if normalized_orientation is None:
            return None, 'Invalid orientation. Use "White" or "Black".'

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Could not decode image data."
        image_height, image_width = image.shape[:2]
        print(f"[MEM] after decode ({image_width}x{image_height}): {_rss_mb():.1f} MB", flush=True)

        largest_side = max(image_height, image_width)
        if largest_side > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(largest_side)
            image = cv2.resize(
                image,
                (int(image_width * scale), int(image_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            print(f"[MEM] after resize: {_rss_mb():.1f} MB", flush=True)

        # 1. Detect board
        match_coords = self.detector.detect(image)
        print(f"[MEM] after detection: {_rss_mb():.1f} MB  detected={match_coords is not None}", flush=True)
        if not match_coords:
            return None, "Could not find a chessboard in the image."

        # 2. Preprocess
        top_left, bottom_right = match_coords
        cropped_board = preprocessing.crop_chessboard(image, top_left, bottom_right)
        squares = preprocessing.divide_and_resize_squares(cropped_board)
        print(f"[MEM] after square split ({squares.shape}): {_rss_mb():.1f} MB", flush=True)

        # 3. Predict
        piece_labels = self.predictor.predict(squares)
        print(f"[MEM] after prediction: {_rss_mb():.1f} MB", flush=True)

        # 4. Convert to FEN
        fen_string = self._convert_to_fen(piece_labels, orientation=normalized_orientation)

        # 5. Format response
        result = {"fen": fen_string}
        if include_cropped_image:
            ok, buffer = cv2.imencode(".jpg", cropped_board, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None, "Could not encode cropped board image."
            b64_image = base64.b64encode(buffer).decode("utf-8")
            result["cropped_image"] = f"data:image/jpeg;base64,{b64_image}"

        print(f"[MEM] request done ({time.perf_counter()-t0:.2f}s): {_rss_mb():.1f} MB", flush=True)
        return result, None
