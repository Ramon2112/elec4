import cv2
import numpy as np

DELIMITER = "#####"

# ტექ Convert text to binary
def text_to_binary(text):
    return ''.join(format(ord(i), '08b') for i in text)

# 🔓 Convert binary to text
def binary_to_text(binary_data):
    chars = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    text = ''.join(chr(int(char, 2)) for char in chars)
    return text

# 🔐 Encode message into image
def encode_image(image_path, secret_message, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    secret_message += DELIMITER
    binary_secret = text_to_binary(secret_message)
    data_index = 0
    data_len = len(binary_secret)

    for row in img:
        for pixel in row:
            for i in range(3):  # R, G, B channels
                if data_index < data_len:
                    pixel[i] = int(format(pixel[i], '08b')[:-1] + binary_secret[data_index], 2)
                    data_index += 1

    cv2.imwrite(output_path, img)
    print("✅ Message encoded successfully!")

# 🔓 Decode message from image
def decode_image(image_path):
    img = cv2.imread(image_path)
    binary_data = ""

    for row in img:
        for pixel in row:
            for i in range(3):
                binary_data += format(pixel[i], '08b')[-1]

    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    decoded_text = ""

    for byte in all_bytes:
        decoded_text += chr(int(byte, 2))
        if DELIMITER in decoded_text:
            break

    return decoded_text.replace(DELIMITER, "")

# ================== MAIN PROGRAM ==================

print("1. Encode Message")
print("2. Decode Message")
choice = input("Enter choice (1/2): ")

if choice == '1':
    image_path = input("Enter path of original image: ")
    message = input("Enter secret message: ")
    output_path = input("Enter output image path: ")
    encode_image(image_path, message, output_path)

elif choice == '2':
    image_path = input("Enter path of stego image: ")
    hidden_message = decode_image(image_path)
    print("🔎 Hidden Message:", hidden_message)

else:
    print("Invalid choice!")
