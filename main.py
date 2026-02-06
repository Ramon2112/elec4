import cv2
from fer import FER
import numpy as np
import os

#para sa camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = FER(mtcnn=True)

# emoji
emoji_folder = "emojis"
emoji_dict = {}
for emo in ["happy","angry","sad","surprise","neutral","disgust","fear"]:
    path = os.path.join(emoji_folder, f"{emo}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # load with alpha channel
    emoji_dict[emo] = img

def overlay_emoji(frame, emoji_img, x, y, w, h):
    # size emoji
    emoji_resized = cv2.resize(emoji_img, (w, h))
    
    if emoji_resized.shape[2] == 4:
        alpha_emoji = emoji_resized[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha_emoji * emoji_resized[:, :, c] +
                                      (1-alpha_emoji) * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = emoji_resized

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = []
    try:
        results = detector.detect_emotions(frame)
    except ValueError:
        pass

    for face in results:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Overlay emoji
        if top_emotion in emoji_dict:
            overlay_emoji(frame, emoji_dict[top_emotion], x, y-50, w, h)

    cv2.imshow("Emoji Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
