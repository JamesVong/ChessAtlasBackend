import os

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'chess_atlas_v1.onnx')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_320.onnx')
DETECTOR_INPUT_SIZE = 320

# Model and Image settings
SQUARE_SIZE = (224, 224)
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))
NORM_MEAN = [0.7675888911192418, 0.658705861886976, 0.528112201226969]
NORM_STD = [0.19983789017033135, 0.20000451458425847, 0.1923135736750554]
LABEL_MAPPING = {
    0: "BlackBishop", 1: "BlackKing", 2: "BlackKnight", 3: "BlackPawn", 4: "BlackQueen", 5: "BlackRook",
    6: "Empty",
    7: "WhiteBishop", 8: "WhiteKing", 9: "WhiteKnight", 10: "WhitePawn", 11: "WhiteQueen", 12: "WhiteRook"
}
PIECE_TO_FEN = {
    "WhiteRook": "R", "WhiteKnight": "N", "WhiteBishop": "B", "WhiteQueen": "Q", "WhiteKing": "K", "WhitePawn": "P",
    "BlackRook": "r", "BlackKnight": "n", "BlackBishop": "b", "BlackQueen": "q", "BlackKing": "k", "BlackPawn": "p",
    "Empty": "1"
}
