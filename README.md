# Image Processing & Object Detection (Flask)

A Flask backend for image editing and object detection: upload an image, crop/flip/rotate/filter it, or run object detection on it using a pretrained SSD MobileNet (TensorFlow) model via OpenCV's DNN module.

## Features

- Image upload/download
- Crop, flip, rotate, and apply filters
- Object detection (SSD MobileNet v3, COCO-trained, via `cv2.dnn`)
- Image feature detection and search
- MySQL-backed image records

## Tech stack

Flask, Flask-CORS, Flask-MySQLdb, OpenCV (`cv2.dnn`), Pillow, matplotlib, MySQL

## Getting started

Requires a running MySQL server.

```bash
cd Flask_project
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env   # then fill in your own DB credentials

python app.py
```

## Environment variables

See [`Flask_project/.env.example`](Flask_project/.env.example) — `SECRET_KEY` and MySQL connection details (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_PORT`). Never commit a real `.env` — it's gitignored.

## Project structure

```
Flask_project/
├── app.py                                          # routes: upload, edit, filter, crop, flip, rotate, objectDetection, featuredetection, search
├── frozen_inference_graph.pb                       # SSD MobileNet v3 (TensorFlow) weights
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt    # model config
├── objectname.txt                                  # COCO class labels
├── static/upload/, static/download/                # sample images / processed output
└── templates/                                       # upload.html, result.html
```

## License

MIT
