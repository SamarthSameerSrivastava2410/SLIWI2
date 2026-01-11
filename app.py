from flask import Flask, Response, render_template, request, jsonify
import cv2 as cv
import numpy as np
import copy
import itertools
from collections import deque, Counter
import mediapipe as mp
import csv
from utils import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

# ---------------- CONFIG ----------------
HISTORY_LENGTH = 16
STATIC_CONF_TH = 0.6
DYNAMIC_CONF_TH = 0.6
KEYPOINT_CSV = "model/keypoint_classifier/keypoint.csv"
POINT_HISTORY_CSV = "model/point_history_classifier/point_history.csv"

# ---------------- Load label CSVs ----------------
def load_labels(csv_path):
    labels = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row:
                labels.append(row[0])
    return labels

KEYPOINT_LABELS = load_labels("model/keypoint_classifier/keypoint_classifier_label.csv")
POINT_HISTORY_LABELS = load_labels("model/point_history_classifier/point_history_classifier_label.csv")

# ---------------- App ----------------
app = Flask(__name__)
cap = None

# ---------------- Mediapipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ---------------- Models ----------------
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# ---------------- State ----------------
cvFpsCalc = CvFpsCalc(buffer_len=10)
point_history = deque(maxlen=HISTORY_LENGTH)
static_history = deque(maxlen=10)
dynamic_history = deque(maxlen=10)
recording_label = None
point_history = deque(maxlen=HISTORY_LENGTH)
static_history = deque(maxlen=10)
dynamic_history = deque(maxlen=10)
recording_label = None

# ---------------- Utils ----------------
def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]

def pre_process_landmark(landmarks):
    temp = copy.deepcopy(landmarks)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] -= base_x
        p[1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp))
    return [v / max_val for v in temp] if max_val != 0 else temp

def pre_process_point_history(history, image):
    h, w = image.shape[:2]
    temp = copy.deepcopy(history)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] = (p[0] - base_x) / w
        p[1] = (p[1] - base_y) / h
    return list(itertools.chain.from_iterable(temp))

def majority_vote(history):
    if not history:
        return -1
    return Counter(history).most_common(1)[0][0]

# ---------------- Flask recording routes ----------------
@app.route("/start_record")
def start_record():
    global recording_label
    label = request.args.get("label", "").upper()
    if label and len(label) == 1 and 'A' <= label <= 'Z':
        recording_label = label
        print(f"[REC] Recording started for label: {recording_label}")
        return jsonify({"status": "recording", "label": recording_label})
    return jsonify({"status": "error", "message": "Invalid label"})

@app.route("/stop_record")
def stop_record():
    global recording_label
    recording_label = None
    print("[REC] Recording stopped")
    return jsonify({"status": "stopped"})

# ---------------- Video streaming ----------------
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    global cap, recording_label, point_history, static_history, dynamic_history
    if cap is None:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(error_frame, "Camera not available", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        ret, buffer = cv.imencode(".jpg", error_frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        debug_image = frame.copy()
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)
        static_label = ""
        dynamic_label = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = calc_landmark_list(frame, hand_landmarks)
            norm_landmarks = pre_process_landmark(landmarks)

            # ---- STATIC ----
            static_id, static_conf = keypoint_classifier(norm_landmarks)
            if static_conf > STATIC_CONF_TH:
                static_history.append(static_id)
            static_result = majority_vote(static_history)
            if static_result != -1 and static_result < len(KEYPOINT_LABELS):
                static_label = KEYPOINT_LABELS[static_result]

            # ---- DYNAMIC ----
            point_history.append(landmarks[8])
            if len(point_history) == HISTORY_LENGTH:
                norm_history = pre_process_point_history(point_history, frame)
                dyn_id, dyn_conf = point_history_classifier(norm_history)
                if dyn_conf > DYNAMIC_CONF_TH:
                    dynamic_history.append(dyn_id)
            dynamic_result = majority_vote(dynamic_history)
            if dynamic_result != -1 and dynamic_result < len(POINT_HISTORY_LABELS):
                dynamic_label = POINT_HISTORY_LABELS[dynamic_result]

            # ---- RECORD DATA ----
            if recording_label:
                with open(KEYPOINT_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(norm_landmarks + [recording_label])
                if len(point_history) == HISTORY_LENGTH:
                    with open(POINT_HISTORY_CSV, "a", newline="") as f:
                        csv.writer(f).writerow(norm_history + [recording_label])

            # ---- DRAW HAND ----
            for lm in landmarks:
                cv.circle(debug_image, (lm[0], lm[1]), 4, (255,255,255), -1)
                cv.circle(debug_image, (lm[0], lm[1]), 4, (0,0,0), 1)

        # ---- UI ----
        fps = cvFpsCalc.get()
        cv.putText(debug_image, f"FPS: {fps:.1f}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if static_label:
            cv.putText(debug_image, f"STATIC: {static_label}", (10,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        if dynamic_label:
            cv.putText(debug_image, f"DYNAMIC: {dynamic_label}", (10,110), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        if recording_label:
            cv.putText(debug_image, f"RECORDING: {recording_label}", (10,150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        ret, buffer = cv.imencode(".jpg", debug_image)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# ---------------- Home ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hand-gestures")
def hand_gestures():
    return render_template("hand-gestures.html")

@app.route("/hand-gestures.html")
def hand_gestures_html():
    return render_template("hand-gestures.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=False)
