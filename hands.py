	import cv2
import mediapipe as mp
from pathlib import Path
from urllib.request import urlretrieve
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
MODEL_URL = (
"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

def _ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
    urlretrieve(MODEL_URL, MODEL_PATH)

def _draw_landmarks_bgr(image_bgr, hand_landmarks) -> None:
    height, width = image_bgr.shape[:2]
    for lm in hand_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image_bgr, (x, y), 2, (0, 255, 0), -1)


def main() -> None:
    _ensure_model()

    base_options = mp_python.BaseOptions(
        model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (camera_id=0)")

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=frame_rgb)
            result = landmarker.detect(mp_image)
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    _draw_landmarks_bgr(frame_bgr, hand_landmarks)

            cv2.imshow("MediaPipe Hands (Tasks)", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

#pip install  opencv-python mediapipe