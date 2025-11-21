"""
Collect hand landmark data using MediaPipe and OpenCV.

Usage:
  python collect_data.py

Controls while running:
 - Press a letter key (A-Z) to set the current label
 - Press 'r' to toggle recording for the current label (records frames to CSV)
 - Press 'q' to quit

Output: data.csv in the current directory with columns: label, x1,y1,z1, x2,y2,z2, ...
"""
import csv
import os
import time
import argparse

import cv2
import mediapipe as mp
import numpy as np


def main(output_file: str):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    recording = False
    current_label = None
    print("Press a letter key to pick label; press 'r' to toggle recording; 'q' to quit")

    # ensure CSV exists and has header
    header = ["label"] + [f"x{i+1}" for i in range(21)] + [f"y{i+1}" for i in range(21)] + [f"z{i+1}" for i in range(21)]
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

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

            # collect landmarks
            lm = []
            h, w, _ = img.shape
            for p in hand.landmark:
                lm.append(p.x)
            for p in hand.landmark:
                lm.append(p.y)
            for p in hand.landmark:
                lm.append(p.z)

            if recording and current_label is not None:
                with open(output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([current_label] + lm)

        # HUD
        status = f"Label: {current_label} | Recording: {recording}"
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if recording else (0, 0, 255), 2)
        cv2.imshow("Collect Data - Press q to quit", img)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            ch = chr(key).upper()
            if ch == 'Q':
                break
            if ch == 'R':
                recording = not recording
                print("Recording:", recording)
            # letter keys
            if ch.isalpha() and len(ch) == 1:
                current_label = ch
                print("Current label set to", current_label)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data.csv', help='Output CSV file')
    args = parser.parse_args()
    main(args.output)
