import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

import time
from pathlib import Path
from ultralytics import YOLO
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Resolve model paths relative to this file so the project is portable
# (works no matter where the repo is cloned or which directory you run it from).
BASE_DIR = Path(__file__).resolve().parent
model = str(BASE_DIR / "models" / "pose_landmarker_heavy.task")
YOLO_model = YOLO(str(BASE_DIR / "models" / "best_dumb_bell.pt"))
latest_result = None

curl_threshold = False
curl_counter = 0


def draw_landmarks_on_image(image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


def compute_angle(vector1, vector2):
  mag_vec_1 = np.linalg.norm(vector1)
  mag_vec_2 = np.linalg.norm(vector2)

  if mag_vec_1 == 0 or mag_vec_2 == 0:
    return 0.0

  dot_product = np.dot(vector1, vector2)

  cos_angle = np.clip(dot_product / (mag_vec_1 * mag_vec_2), -1.0, 1.0)
  angle = np.degrees(np.arccos(cos_angle))
  return angle


def result_callback(result, output_image: mp.Image, timestamp_ms: int):
  global latest_result
  latest_result = result


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback)

with PoseLandmarker.create_from_options(options) as detector:

  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print("Failed to open camera")
    exit()

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Failed to read frame")
      break
    h, w = frame.shape[:2]

    # Detect dumbbells. In detection mode results[0].boxes is ALWAYS a Boxes
    # object (never None), so test how many boxes were found, not None-ness.
    yolo_results = YOLO_model(frame, verbose=False)[0]

    if len(yolo_results.boxes) > 0:
      for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Label sits just above the top edge of the box (kept on-screen).
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
        cv2.putText(frame, "Dumbell Detected", (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    else:
      # No dumbbell -> message in the bottom-left corner of the frame.
      cv2.putText(frame, "No Dumbell Detected", (15, h - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detector.detect_async(mp_frame, int(time.time() * 1000))

    result = latest_result

    if result is not None and result.pose_landmarks and result.pose_world_landmarks:
      frame = draw_landmarks_on_image(frame, result)

      landmarks = result.pose_landmarks[0]
      world = result.pose_world_landmarks[0]

      right_shoulder = landmarks[12]
      right_elbow = landmarks[14]
      right_wrist = landmarks[16]

      elbow_to_shoulder = np.array([
          right_shoulder.x - right_elbow.x,
          right_shoulder.y - right_elbow.y,
          ])
      elbow_to_wrist = np.array([
          right_wrist.x - right_elbow.x,
          right_wrist.y - right_elbow.y,
          ])

      required_angle = compute_angle(elbow_to_shoulder, elbow_to_wrist)

      if required_angle > 160:
        curl_threshold = False
      if required_angle < 30 and not curl_threshold:
        curl_threshold = True
        curl_counter += 1

      elbow_px = (int(landmarks[14].x * w), int(landmarks[14].y * h))
      cv2.putText(frame, f"{int(required_angle)} deg",
                  (elbow_px[0] + 10, elbow_px[1]),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (230, 75), (245, 117, 16), -1)
    cv2.putText(frame, "CURLS", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str(curl_counter), (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    stage = "up" if curl_threshold else "down"
    cv2.putText(frame, f"stage: {stage}", (110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Curl Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
