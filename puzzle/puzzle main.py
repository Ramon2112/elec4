import cv2
import numpy as np
import random

def create_jigsaw_puzzle(image_path, grid_size, output_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    height, width, _ = img.shape

    # Resize image so it's divisible by grid size
    new_height = height - (height % grid_size)
    new_width = width - (width % grid_size)
    img = cv2.resize(img, (new_width, new_height))

    piece_h = new_height // grid_size
    piece_w = new_width // grid_size

    pieces = []

    # Split image into pieces
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * piece_h
            y2 = y1 + piece_h
            x1 = col * piece_w
            x2 = x1 + piece_w

            piece = img[y1:y2, x1:x2]
            pieces.append(piece)

    # Shuffle the pieces
    random.shuffle(pieces)

    # Create blank canvas for puzzle image
    puzzle_img = np.zeros_like(img)

    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * piece_h
            y2 = y1 + piece_h
            x1 = col * piece_w
            x2 = x1 + piece_w

            puzzle_img[y1:y2, x1:x2] = pieces[index]
            index += 1

    # Save and show result
    cv2.imwrite(output_path, puzzle_img)
    cv2.imshow("Jigsaw Puzzle", puzzle_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================== RUN PROGRAM ==================

image_path = input("Enter path of image: ")
grid_size = int(input("Enter grid size (e.g., 3 for 3x3): "))
output_path = input("Enter output image name: ")

create_jigsaw_puzzle(image_path, grid_size, output_path)
