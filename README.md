# Face Recognition OpenCV

A Python-based face recognition system built with OpenCV, with a Flask web app for browser-based face detection and recognition.

## Features

- Detects and recognizes faces from images/webcam using OpenCV
- Stores known faces for comparison in the `known_faces/` directory
- Flask web interface for uploading images or streaming from a webcam
- Simple, lightweight setup — no heavy ML framework required

## Project Structure

```
Face_Recognition_OpenCv/
├── known_faces/            # Reference images of known people
├── face_rec.py              # Core face recognition script (CLI/local)
├── flask-app_face_rec.py    # Flask web app wrapper
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- `face_recognition` (if used) or `numpy`
- Flask (for the web app)

Install dependencies:

```bash
pip install opencv-python face_recognition flask numpy
```

> If you're using a `requirements.txt`, replace the above with:
> ```bash
> pip install -r requirements.txt
> ```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aicp8921/Face_Recognition_OpenCv.git
   cd Face_Recognition_OpenCv
   ```

2. Add reference photos of people you want to recognize into the `known_faces/` folder. Name each file after the person, e.g. `known_faces/john_doe.jpg`.

3. Run the script or the Flask app (see below).

## Usage

### Run locally (script mode)

```bash
python face_rec.py
```

This will open your webcam (or process a given image, depending on your script's configuration) and identify faces against the `known_faces/` directory.

### Run the Flask web app

```bash
python flask-app_face_rec.py
```

Then open your browser at:

```
http://127.0.0.1:5000
```

Upload an image or use the live feed to detect and recognize faces.

## How It Works

1. Known faces are loaded from `known_faces/` and encoded.
2. Incoming faces (from an image or webcam frame) are detected using OpenCV's face detection.
3. Detected faces are compared against the known encodings.
4. Matches are labeled with the corresponding name; unrecognized faces are labeled as "Unknown."

## Roadmap / Ideas

- [ ] Add support for multiple camera sources
- [ ] Improve recognition accuracy with a deep learning model
- [ ] Add a database to store recognition logs
- [ ] Dockerize the app for easier deployment

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## License

Specify your license here (e.g., MIT, Apache 2.0).
