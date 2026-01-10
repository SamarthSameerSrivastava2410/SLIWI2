print("🔥 ACTUAL point_history_classifier.py USED:", __file__)

import os
import numpy as np
import tensorflow as tf

class PointHistoryClassifier(object):
    def __init__(self):
        print(">> Initializing TFLite PointHistoryClassifier")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            BASE_DIR,
            "point_history_classifier.tflite"
        )

        print("Model path:", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ MODEL FILE NOT FOUND: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(">> PointHistoryClassifier READY")

    def __call__(self, point_history):
        input_data = np.array([point_history], dtype=np.float32)
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_data
        )
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        return np.argmax(result)
