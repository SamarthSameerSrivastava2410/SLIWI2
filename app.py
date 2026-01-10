import csv
import copy
import itertools
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from flask import Flask, Response, render_template

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
from utils import CvFpsCalc

# git add .
# git commit -a -m "<message>"
# git push 

# ---------------- FLASK APP ---------------- #
app = Flask(__name__)


# ---------------- CAMERA ---------------- #
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)


# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


# ---------------- CLASSIFIERS ---------------- #
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

with open(
    'model/keypoint_classifier/keypoint_classifier_label.csv',
    encoding='utf-8-sig'
) as f:
    keypoint_labels = [row[0] for row in csv.reader(f)]

with open(
    'model/point_history_classifier/point_history_classifier_label.csv',
    encoding='utf-8-sig'
) as f:
    point_history_labels = [row[0] for row in csv.reader(f)]


# ---------------- STATE ---------------- #
cvFpsCalc = CvFpsCalc(buffer_len=10)
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)
mode = 0
number = -1


# ---------------- DRAW FUNCTIONS ---------------- #
def draw_landmarks(image, landmark_point):
    for point in landmark_point:
        cv.circle(image, (point[0], point[1]), 5, (255, 255, 255), -1)
        cv.circle(image, (point[0], point[1]), 5, (0, 0, 0), 1)
    return image


def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]),
                 (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]-22),
                 (brect[2], brect[1]), (0, 0, 0), -1)
    text = handedness.classification[0].label
    if hand_sign_text:
        text += ":" + hand_sign_text
    cv.putText(image, text, (brect[0]+5, brect[1]-4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if finger_gesture_text:
        cv.putText(image, "Finger Gesture: " + finger_gesture_text,
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def draw_point_history(image, point_history):
    for i, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]),
                      1 + int(i / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps):
    cv.putText(image, f"FPS:{fps}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


# ---------------- PROCESS FUNCTIONS ---------------- #
def calc_bounding_rect(image, landmarks):
    h, w = image.shape[:2]
    points = []
    for lm in landmarks.landmark:
        points.append([int(lm.x * w), int(lm.y * h)])
    x, y, w, h = cv.boundingRect(np.array(points))
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] -= base_x
        p[1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(map(abs, temp))
    return [v / max_value for v in temp]


def pre_process_point_history(point_history, image):
    if not point_history:
        return []
    h, w = image.shape[:2]
    temp = copy.deepcopy(point_history)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] = (p[0] - base_x) / w
        p[1] = (p[1] - base_y) / h
    return list(itertools.chain.from_iterable(temp))


# ---------------- VIDEO STREAM ---------------- #
def generate_frames():
    global point_history, finger_gesture_history

    while True:
        fps = cvFpsCalc.get()
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmarks = calc_landmark_list(debug_image, hand_landmarks)

                pre_landmark = pre_process_landmark(landmarks)
                pre_history = pre_process_point_history(point_history, debug_image)

                hand_id = keypoint_classifier(pre_landmark)

                if hand_id == 2:
                    point_history.append(landmarks[8])
                else:
                    point_history.append([0, 0])

                gesture_id = 0
                if len(pre_history) == history_length * 2:
                    gesture_id = point_history_classifier(pre_history)

                finger_gesture_history.append(gesture_id)
                gesture = Counter(finger_gesture_history).most_common(1)[0][0]

                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmarks)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_labels[hand_id],
                    point_history_labels[gesture]
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps)

        ret, buffer = cv.imencode('.jpg', debug_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ---------------- #
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    app.run(debug=True)
