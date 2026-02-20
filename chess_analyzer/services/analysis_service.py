import cv2
import numpy as np
import base64
from config import MAX_IMAGE_DIM, PIECE_TO_FEN
from chess_analyzer.vision import preprocessing

class ChessAnalysisService:
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

    def analyze_image(self, image_bytes, include_cropped_image=True):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Could not decode image data."
        image_height, image_width = image.shape[:2]
        largest_side = max(image_height, image_width)
        if largest_side > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(largest_side)
            image = cv2.resize(
                image,
                (int(image_width * scale), int(image_height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # 1. Detect board
        match_coords = self.detector.detect(image)
        if not match_coords:
            return None, "Could not find a chessboard in the image."
        
        # 2. Preprocess
        top_left, bottom_right = match_coords
        cropped_board = preprocessing.crop_chessboard(image, top_left, bottom_right)
        squares = preprocessing.divide_and_resize_squares(cropped_board)
        
        # 3. Predict
        piece_labels = self.predictor.predict(squares)
        
        # 4. Convert to FEN
        fen_string = self._convert_to_fen(piece_labels)
        
        # 5. Format response
        result = {"fen": fen_string}
        if include_cropped_image:
            ok, buffer = cv2.imencode(".jpg", cropped_board, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None, "Could not encode cropped board image."
            b64_image = base64.b64encode(buffer).decode("utf-8")
            result["cropped_image"] = f"data:image/jpeg;base64,{b64_image}"

        return result, None
