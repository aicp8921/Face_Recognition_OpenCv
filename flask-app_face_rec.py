import cv2
import os
import numpy as np
from flask import Flask, render_template_string, Response

app = Flask(__name__)

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
        aligned_face = recognizer.alignCrop(img, faces[0])
        feature = recognizer.feature(aligned_face)
        return feature
    return None

# 4. Load & Encode Known Faces at Startup
known_features = []
known_names = []

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

for file_name in os.listdir(KNOWN_FACES_DIR):
    if file_name.endswith(('.jpg', '.png', '.jpeg')):
        person_name = os.path.splitext(file_name)[0].capitalize()
        img_path = os.path.join(KNOWN_FACES_DIR, file_name)
        feat = get_face_feature(img_path)
        if feat is not None:
            known_features.append(feat)
            known_names.append(person_name)
            print(f"[INFO] Successfully encoded: {person_name}")

COSINE_THRESHOLD = 0.36

# 5. Video Stream Generator Function
def generate_frames():
    # Open camera inside the generator to ensure proper release on stop
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        detector.setInputSize((w, h))
        
        _, faces = detector.detect(frame)
        if faces is not None:
            for face in faces:
                box = list(map(int, face[:4]))
                aligned_face = recognizer.alignCrop(frame, face)
                curr_feature = recognizer.feature(aligned_face)
                
                name = "Unknown"
                max_score = 0.0
                
                for known_feat, known_name in zip(known_features, known_names):
                    score = recognizer.match(known_feat, curr_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > COSINE_THRESHOLD and score > max_score:
                        max_score = score
                        name = f"{known_name} ({int(score * 100)}%)"
                
                x, y, width, height = box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Encode the processed frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        # Yield the output frame in the format expected by multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

# 6. Flask Web Routes
@app.route('/')
def index():
    # Simple HTML template embedding the live video feed
    html_page = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition Stream</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f2f5; margin-top: 50px; }
            h1 { color: #333; }
            .video-container { margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: inline-block; border-radius: 8px; overflow: hidden; }
        </style>
    </head>
    <body>
        <h1>OpenCV Face Recognition</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
    </body>
    </html>
    """
    return render_template_string(html_page)

@app.route('/video_feed')
def video_feed():
    # Return the stream response with the correct multipart content type
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on local network
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
