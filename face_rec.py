import cv2
import os
import numpy as np

# 1. Model File Paths
DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "known_faces"

# 2. Initialize OpenCV 4 DNN Modules
detector = cv2.FaceDetectorYN.create(
    model=DETECTOR_MODEL,
    config="",
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)

recognizer = cv2.FaceRecognizerSF.create(
    model=RECOGNIZER_MODEL,
    config=""
)

# 3. Helper Function to Get Feature Embeddings
def get_face_feature(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w, _ = img.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    if faces is not None:
        # Align and crop the first detected face
        aligned_face = recognizer.alignCrop(img, faces[0])
        # Extract 128-d facial feature vector
        feature = recognizer.feature(aligned_face)
        return feature
    return None

# 4. Load & Encode Known Faces
known_features = []
known_names = []

for file_name in os.listdir(KNOWN_FACES_DIR):
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
        person_name = os.path.splitext(file_name)[0].capitalize()
        img_path = os.path.join(KNOWN_FACES_DIR, file_name)
        feat = get_face_feature(img_path)
        
        if feat is not None:
            known_features.append(feat)
            known_names.append(person_name)
            print(f"[INFO] Successfully encoded: {person_name}")

# 5. Real-Time Camera Stream Recognition
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Match Thresholds for SFace:
# Cosine Similarity >= 0.36 or L2 Distance <= 1.12 indicates a match.
COSINE_THRESHOLD = 0.36

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    # Detect faces in frame
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Extract bounding box coordinates
            box = list(map(int, face[:4]))
            confidence = face[-1]

            # Align face and compute embedding
            aligned_face = recognizer.alignCrop(frame, face)
            curr_feature = recognizer.feature(aligned_face)

            name = "Unknown"
            max_score = 0.0

            # Match against known feature database
            for known_feat, known_name in zip(known_features, known_names):
                # Calculate Cosine similarity score
                score = recognizer.match(known_feat, curr_feature, cv2.FaceRecognizerSF_FR_COSINE)
                
                if score > COSINE_THRESHOLD and score > max_score:
                    max_score = score
                    name = f"{known_name} ({int(score * 100)}%)"

            # Draw results on screen
            x, y, width, height = box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("OpenCV 4 Native Face Recognition (YuNet + SFace)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


"""
Must download the following models from OpenCV Zoo before running this script:---

wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
"""