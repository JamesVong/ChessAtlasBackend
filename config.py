import os

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'chess_piece_classifier_mobilenet.h5')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.onnx')

# Model and Image settings
SQUARE_SIZE = (64, 64)
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))
LABEL_MAPPING = {
    "0": "Black_Bishop", "1": "Black_King", "2": "Black_Knight", "3": "Black_Pawn", "4": "Black_Queen", "5": "Black_Rook",
    "6": "Empty",
    "7": "White_Bishop", "8": "White_King", "9": "White_Knight", "10": "White_Pawn", "11": "White_Queen", "12": "White_Rook"
}
PIECE_TO_FEN = {
    "White_Rook": "R", "White_Knight": "N", "White_Bishop": "B", "White_Queen": "Q", "White_King": "K", "White_Pawn": "P",
    "Black_Rook": "r", "Black_Knight": "n", "Black_Bishop": "b", "Black_Queen": "q", "Black_King": "k", "Black_Pawn": "p",
    "Empty": "1"
}
