import cv2
from config import SQUARE_SIZE

def crop_chessboard(image, top_left, bottom_right):
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

def divide_and_resize_squares(board_image):
    squares = []
    height, width, _ = board_image.shape
    square_height, square_width = height // 8, width // 8
    for row in range(8):
        for col in range(8):
            square = board_image[row*square_height:(row+1)*square_height, col*square_width:(col+1)*square_width]
            resized_square = cv2.resize(square, SQUARE_SIZE, interpolation=cv2.INTER_AREA)
            squares.append(resized_square)
    return squares
