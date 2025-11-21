"""
Real-time recognition using trained model and MediaPipe. Speaks recognized labels via pyttsx3.

Usage:
  python recognize.py --model model.pkl
"""
import argparse
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3


def main(model_path: str):
    data = joblib.load(model_path)
    model = data['model']
    le = data['label_encoder']

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    last_pred = None
    stable_count = 0
    STABLE_THRESHOLD = 6
    output_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            lm = []
            for p in hand.landmark:
                lm.append(p.x)
            for p in hand.landmark:
                lm.append(p.y)
            for p in hand.landmark:
                lm.append(p.z)

            X = np.array(lm).reshape(1, -1)
            pred = model.predict(X)[0]
            label = le.inverse_transform([pred])[0]

            # stability check
            if label == last_pred:
                stable_count += 1
            else:
                stable_count = 0
            last_pred = label

            if stable_count >= STABLE_THRESHOLD:
                # append to output and speak
                output_text += label
                print("Recognized:", label)
                engine.say(label)
                engine.runAndWait()
                stable_count = 0

            cv2.putText(img, f"Pred: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.putText(img, f"Output: {output_text}", (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Sign Recognizer", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            output_text = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.pkl', help='Trained model file')
    args = parser.parse_args()
    main(args.model)
