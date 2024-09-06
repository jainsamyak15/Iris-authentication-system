import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import hamming, euclidean, cosine
import os
import torch


class IrisAuthenticationSystem:
    def __init__(self, eye_model_path, iris_model_path):
        self.eye_model = YOLO(eye_model_path)
        self.iris_model = YOLO(iris_model_path)
        self.templates = {}
        self.feature_size = 10000  # Adjust this based on your needs

    def preprocess_image(self, image):
        if image is None:
            raise ValueError("Input image is None")

        # Ensure image is in BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Convert to RGB (YOLO models typically expect RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to a size that YOLO expects (e.g., 640x640)
        image_resized = cv2.resize(image_rgb, (640, 640))

        return image_resized

    def detect_eyes(self, image):
        try:
            results = self.eye_model(image)
            eyes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    eyes.append((int(x1), int(y1), int(x2), int(y2)))
            print(f"Detected {len(eyes)} eyes")
            return eyes
        except Exception as e:
            print(f"Error in detect_eyes: {str(e)}")
            return []

    def segment_iris(self, eye_image):
        try:
            results = self.iris_model(eye_image)
            for r in results:
                masks = r.masks
                if masks is not None:
                    return masks.data[0].cpu().numpy()
            return None
        except Exception as e:
            print(f"Error in segment_iris: {str(e)}")
            return None

    def extract_features(self, iris_mask):
        try:
            iris_mask_8bit = (iris_mask * 255).astype(np.uint8)

            # Apply Gabor filter
            kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(iris_mask_8bit, -1, kernel)

            # Normalize filtered image to 0-255 range
            filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Resize to a fixed size
            resized = cv2.resize(filtered_norm, (100, 100))

            # Flatten and normalize
            features = resized.flatten() / 255.0

            # Pad or truncate to ensure consistent size
            if len(features) > self.feature_size:
                features = features[:self.feature_size]
            elif len(features) < self.feature_size:
                features = np.pad(features, (0, self.feature_size - len(features)), 'constant')

            return features
        except Exception as e:
            print(f"Error in extract_features: {str(e)}")
            return np.zeros(self.feature_size)

    def register_user(self, user_id, image):
        try:
            preprocessed = self.preprocess_image(image)
            eyes = self.detect_eyes(preprocessed)
            if not eyes:
                return False, "No eyes detected"

            features = []
            for eye in eyes:
                x1, y1, x2, y2 = eye
                eye_image = preprocessed[y1:y2, x1:x2]
                iris_mask = self.segment_iris(eye_image)
                if iris_mask is not None:
                    features.append(self.extract_features(iris_mask))

            if not features:
                return False, "Failed to extract iris features"

            self.templates[user_id] = features
            return True, "User registered successfully"
        except Exception as e:
            return False, f"Error during registration: {str(e)}"

    def match_templates(self, test_template, stored_template):
        try:
            # Ensure templates are of the same size
            min_shape = min(len(test_template), len(stored_template))
            test_template = test_template[:min_shape]
            stored_template = stored_template[:min_shape]

            # Calculate various similarity metrics
            hamming_dist = hamming(test_template, stored_template)
            euclidean_dist = euclidean(test_template, stored_template)
            cosine_sim = 1 - cosine(test_template, stored_template)

            return hamming_dist, euclidean_dist, cosine_sim
        except Exception as e:
            print(f"Error in match_templates: {str(e)}")
            return 1.0, float('inf'), 0.0  # Worst case values

    def authenticate(self, user_id, image):
        try:
            if user_id not in self.templates:
                return False, "User not registered"

            preprocessed = self.preprocess_image(image)
            eyes = self.detect_eyes(preprocessed)
            if not eyes:
                return False, "No eyes detected"

            test_features = []
            for eye in eyes:
                x1, y1, x2, y2 = eye
                eye_image = preprocessed[y1:y2, x1:x2]
                iris_mask = self.segment_iris(eye_image)
                if iris_mask is not None:
                    test_features.append(self.extract_features(iris_mask))

            if not test_features:
                return False, "Failed to extract iris features"

            stored_features = self.templates[user_id]

            best_hamming = float('inf')
            best_euclidean = float('inf')
            best_cosine = 0

            for test_feature in test_features:
                for stored_feature in stored_features:
                    hamming_dist, euclidean_dist, cosine_sim = self.match_templates(test_feature, stored_feature)
                    best_hamming = min(best_hamming, hamming_dist)
                    best_euclidean = min(best_euclidean, euclidean_dist)
                    best_cosine = max(best_cosine, cosine_sim)

            print(f"Best Hamming distance: {best_hamming}")
            print(f"Best Euclidean distance: {best_euclidean}")
            print(f"Best Cosine similarity: {best_cosine}")

            # Set thresholds for each metric
            hamming_threshold = 0.2
            euclidean_threshold = 0.3
            cosine_threshold = 0.8

            # Initialize a counter for passed metrics
            passed_metrics = 0

            # Check each condition and increment the counter if it passes
            if best_hamming < hamming_threshold:
                passed_metrics += 1
            if best_euclidean < euclidean_threshold:
                passed_metrics += 1
            if best_cosine > cosine_threshold:  # Assuming higher cosine similarity is better
                passed_metrics += 1

            # Authentication is successful if at least two conditions are met
            if passed_metrics >= 2:
                return True, "Authentication successful"
            else:
                return False, "Authentication failed"


        except Exception as e:
            return False, f"Error during authentication: {str(e)}"
