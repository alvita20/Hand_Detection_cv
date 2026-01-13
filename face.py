import cv2
import mediapipe as mp

from pathlib import Path
from urllib.request import urlretrieve

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")


def _ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"Downloading face landmarker model to {MODEL_PATH} ...")
    urlretrieve(MODEL_URL, MODEL_PATH)


def _draw_landmarks_bgr(image_bgr, face_landmarks) -> None:
    height, width = image_bgr.shape[:2]
    for lm in face_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image_bgr, (x, y), 1, (200, 105, 20), -1)


def main() -> None:
    _ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (camera_id=0)")

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    _draw_landmarks_bgr(frame_bgr, face_landmarks)

            cv2.imshow("MediaPipe Face Landmarks (Tasks)", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()