import csv
import numpy as np
import tensorflow as tf
import os

print(">> Initializing TFLite KeyPointClassifier")

class KeyPointClassifier:
    def __init__(self, model_path="model/keypoint_classifier/keypoint_classifier.tflite"):
        self.model_path = model_path
        if not os.path.isfile(self.model_path):
            print(f"❌ MODEL FILE NOT FOUND: {self.model_path}")
        else:
            print(f"✅ Loading model from: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded successfully")

        self.num_classes = 21  # change according to your model

    def __call__(self, landmark_list):
        if not hasattr(self, "interpreter"):
            return -1  # model not loaded
        # Input must be float32 and shaped like (1, 63) for 21 landmarks * 3
        input_data = np.array(landmark_list, dtype=np.float32).reshape(1, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return int(np.argmax(output_data))
