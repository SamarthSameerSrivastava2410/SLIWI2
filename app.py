from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
from collections import deque, Counter

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ================= CONFIG =================
HISTORY_LENGTH = 16
STATIC_CONF_TH = 0.6
DYNAMIC_CONF_TH = 0.6

KEYPOINT_CSV = "model/keypoint_classifier/keypoint.csv"
POINT_HISTORY_CSV = "model/point_history_classifier/point_history.csv"

KEYPOINT_LABEL_CSV = "model/keypoint_classifier/keypoint_classifier_label.csv"
POINT_HISTORY_LABEL_CSV = "model/point_history_classifier/point_history_classifier_label.csv"

# ================= LOAD LABELS =================
def load_labels(path):
    labels = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            labels.append(row[0])
    return labels

KEYPOINT_LABELS = load_labels(KEYPOINT_LABEL_CSV)
POINT_HISTORY_LABELS = load_labels(POINT_HISTORY_LABEL_CSV)

# ================= HELPERS =================
def majority_vote(history):
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]

def calc_landmarks(image, landmarks):
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
    return [v / max_val for v in temp] if max_val else temp

def pre_process_point_history(history, image):
    h, w = image.shape[:2]
    temp = copy.deepcopy(history)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] = (p[0] - base_x) / w
        p[1] = (p[1] - base_y) / h
    return list(itertools.chain.from_iterable(temp))

# ================= APP =================
app = Flask(__name__)
cap = cv2.VideoCapture(0)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ================= MODELS =================
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# ================= STATE =================
point_history = deque(maxlen=HISTORY_LENGTH)
static_history = deque(maxlen=10)
dynamic_history = deque(maxlen=10)
recording_label = None
recording_mode = None   # "static" or "dynamic"

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_record", methods=["POST"])
def start_record():
    global recording_label, recording_mode
    data = request.json
    recording_label = data["label"]
    recording_mode = data["mode"]
    return jsonify({"status": "recording"})

@app.route("/stop_record", methods=["POST"])
def stop_record():
    global recording_label, recording_mode
    recording_label = None
    recording_mode = None
    return jsonify({"status": "stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ================= VIDEO LOOP =================
def generate_frames():
    global recording_label, recording_mode

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ---------- DEFAULT VALUES (VERY IMPORTANT) ----------
        static_result = None
        static_conf = 0.0
        static_label = ""
        dynamic_label = ""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = calc_landmarks(frame, hand_landmarks)

            # ---------- DRAW HAND SKELETON ----------
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # ---------- STATIC ----------
            norm_landmarks = pre_process_landmark(landmarks)
            static_id, static_conf = keypoint_classifier(norm_landmarks)

            if static_conf > STATIC_CONF_TH:
                static_history.append(static_id)

            static_result = majority_vote(static_history)
            if static_result is not None and 0 <= static_result < len(KEYPOINT_LABELS):
                static_label = KEYPOINT_LABELS[static_result]

            # ---------- STATIC CONFIDENCE ----------
            cv2.putText(
                frame,
                f"Conf: {static_conf * 100:.1f}%",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # ---------- DYNAMIC ----------
            point_history.append(landmarks[8])  # index fingertip

            if len(point_history) == HISTORY_LENGTH:
                norm_history = pre_process_point_history(point_history, frame)
                dyn_label, dyn_conf = point_history_classifier(norm_history)

                if dyn_conf > DYNAMIC_CONF_TH:
                    dynamic_history.append(dyn_label)

            dynamic_result = majority_vote(dynamic_history)
            if dynamic_result:
                dynamic_label = dynamic_result

            # ---------- RECORD ----------
            if recording_label:
                if recording_mode == "static":
                    with open(KEYPOINT_CSV, "a", newline="") as f:
                        csv.writer(f).writerow(norm_landmarks + [recording_label])

                elif recording_mode == "dynamic" and len(point_history) == HISTORY_LENGTH:
                    with open(POINT_HISTORY_CSV, "a", newline="") as f:
                        csv.writer(f).writerow(norm_history + [recording_label])

        # ---------- UI ----------
        cv2.putText(frame, f"Static: {static_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Dynamic: {dynamic_label}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if recording_label:
            cv2.putText(frame, f"REC {recording_mode.upper()}: {recording_label}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=False)
