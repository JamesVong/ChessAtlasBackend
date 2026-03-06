import cv2
import numpy as np
from config import SQUARE_SIZE

# Each raw square should be at least this wide/tall before the final resize.
# Below this, the 11x upscale to 224 produces images too far from training distribution.
_MIN_SQUARE_PX = 112
_MIN_BOARD_PX = 8 * _MIN_SQUARE_PX  # 896

def crop_chessboard(image, top_left, bottom_right):
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

def divide_and_resize_squares(board_image):
    height, width, _ = board_image.shape
    if height < 8 or width < 8:
        raise ValueError("Detected chessboard area is too small.")

    # Upscale small boards so each raw square is at least _MIN_SQUARE_PX px.
    # INTER_CUBIC preserves sharpness better than INTER_LINEAR when upscaling.
    if height < _MIN_BOARD_PX or width < _MIN_BOARD_PX:
        scale = _MIN_BOARD_PX / min(height, width)
        board_image = cv2.resize(
            board_image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        height, width = board_image.shape[:2]

    square_height, square_width = height // 8, width // 8
    squares = np.empty((64, SQUARE_SIZE[1], SQUARE_SIZE[0], 3), dtype=np.uint8)
    square_idx = 0
    for row in range(8):
        for col in range(8):
            square = board_image[row*square_height:(row+1)*square_height, col*square_width:(col+1)*square_width]
            # INTER_AREA for downscale, INTER_CUBIC for upscale
            interp = cv2.INTER_AREA if square_height >= SQUARE_SIZE[1] else cv2.INTER_CUBIC
            resized_square = cv2.resize(square, SQUARE_SIZE, interpolation=interp)
            squares[square_idx] = resized_square
            square_idx += 1

    return squares
