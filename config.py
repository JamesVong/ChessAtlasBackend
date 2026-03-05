import os

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'chess_atlas_v1.torchscript.pt')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'templates', 'chessboard_template.png')

# Model and Image settings
SQUARE_SIZE = (224, 224)
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1600"))

# Label index → class name (matches training IDX2LABEL order)
LABEL_MAPPING = [
    "BlackBishop", "BlackKing",   "BlackKnight", "BlackPawn",
    "BlackQueen",  "BlackRook",   "Empty",
    "WhiteBishop", "WhiteKing",   "WhiteKnight", "WhitePawn",
    "WhiteQueen",  "WhiteRook",
]

PIECE_TO_FEN = {
    "WhiteRook": "R", "WhiteKnight": "N", "WhiteBishop": "B",
    "WhiteQueen": "Q", "WhiteKing": "K", "WhitePawn": "P",
    "BlackRook": "r", "BlackKnight": "n", "BlackBishop": "b",
    "BlackQueen": "q", "BlackKing": "k", "BlackPawn": "p",
    "Empty": "1",
}
