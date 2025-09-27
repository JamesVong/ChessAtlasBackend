import cv2
import numpy as np
import base64
from config import PIECE_TO_FEN
from chess_analyzer.vision import detector, preprocessing
from chess_analyzer.ml.predictor import PiecePredictor

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

    def analyze_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
        _, buffer = cv2.imencode('.jpg', cropped_board)
        b64_image = base64.b64encode(buffer).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64_image}"
        
        result = {"fen": fen_string, "cropped_image": data_url}
        return result, None