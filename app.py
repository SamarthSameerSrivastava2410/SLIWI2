import csv
import copy
import itertools
from collections import Counter, deque

import numpy as np
import cv2 as cv
import mediapipe as mp
from flask import Flask, Response, render_template, jsonify

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
from utils import CvFpsCalc

app = Flask(__name__)

# ---------------- Recording state ----------------
recording = False
current_label = -1
next_label = 0  # auto-increment label

# ---------------- Camera ----------------
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ---------------- Mediapipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---------------- Classifiers ----------------
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_labels = [row[0] for row in csv.reader(f)]

with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_labels = [row[0] for row in csv.reader(f)]

# ---------------- State ----------------
cvFpsCalc = CvFpsCalc(buffer_len=10)
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

# ---------------- Drawing functions ----------------
def draw_landmarks(image, landmark_point):
    for point in landmark_point:
        cv.circle(image, (point[0], point[1]), 5, (255, 255, 255), -1)
        cv.circle(image, (point[0], point[1]), 5, (0, 0, 0), 1)
    return image

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_point_history(image, point_history):
    for i, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(i / 2), (152, 251, 152), 2)
    return image

# ---------------- Processing functions ----------------
def calc_bounding_rect(image, landmarks):
    h, w = image.shape[:2]
    points = [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]
    x, y, w, h = cv.boundingRect(np.array(points))
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x*w), int(lm.y*h)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] -= base_x
        p[1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(map(abs,temp))
    return [v/max_value for v in temp]

def pre_process_point_history(point_history_list, image):
    if not point_history_list:
        return []
    h,w = image.shape[:2]
    temp = copy.deepcopy(point_history_list)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] = (p[0]-base_x)/w
        p[1] = (p[1]-base_y)/h
    return list(itertools.chain.from_iterable(temp))

# ---------------- Video stream ----------------
def generate_frames():
    global point_history, finger_gesture_history
    global recording, current_label

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame,1)
        debug_image = frame.copy()

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        try:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = calc_landmark_list(frame, hand_landmarks)
                    pre_landmark = pre_process_landmark(landmarks)
                    pre_history = pre_process_point_history(point_history, frame)

                    # --- Classifier predictions ---
                    hand_id = keypoint_classifier(pre_landmark)
                    gesture_id = 0
                    if len(pre_history) == history_length*2:
                        gesture_id = point_history_classifier(pre_history)

                    finger_gesture_history.append(gesture_id)

                    # --- Point history ---
                    if hand_id == 2:
                        point_history.append(landmarks[8])
                    else:
                        point_history.append([0,0])

                    # --- Record CSV ---
                    if recording and current_label != -1:
                        # static keypoints
                        with open("model/keypoint_classifier/keypoint.csv","a",newline="") as f:
                            csv.writer(f).writerow(pre_landmark + [current_label])
                        # dynamic point history
                        if len(pre_history) == history_length*2:
                            with open("model/point_history_classifier/point_history.csv","a",newline="") as f:
                                csv.writer(f).writerow(pre_history + [current_label])

                    # --- Draw bounding box & landmarks ---
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    debug_image = draw_bounding_rect(debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmarks)

                    # --- Draw predicted label above hand ---
                    hand_id = keypoint_classifier(pre_landmark)  # predicted class index
                    if 0 <= hand_id < len(keypoint_labels):
                        predicted_label = keypoint_labels[hand_id]  # map index to letter
                    else:
                        predicted_label = ""
                    if predicted_label:
                        cv.putText(debug_image, predicted_label, (brect[0], brect[1]-10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)


        except Exception as e:
            print("Error processing hand:", e)

        # --- Draw point history ---
        debug_image = draw_point_history(debug_image, point_history)

        # --- Draw FPS / recording status ---
        fps = cvFpsCalc.get()
        cv.putText(debug_image, f"FPS:{fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if recording and 0 <= current_label < len(keypoint_labels):
            cv.putText(debug_image, f"RECORDING: {keypoint_labels[current_label]}", (10,50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        ret, buffer = cv.imencode('.jpg', debug_image)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_record')
def start_record():
    global recording, current_label, next_label
    recording = True
    current_label = next_label
    next_label += 1
    print(f"Recording started for label {current_label}")
    return jsonify({"status":"recording","label":current_label})

@app.route('/stop_record')
def stop_record():
    global recording
    recording = False
    print("Recording stopped")
    return jsonify({"status":"stopped"})

if __name__ == "__main__":
    app.run(debug=False)
