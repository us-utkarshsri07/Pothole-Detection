
# Deep Learning-Based Pothole Detection

A computer vision application that detects potholes in road images using YOLOv8n.

## Features

- Image upload
- Pothole detection using YOLOv8n
- Bounding-box visualization
- Confidence scores
- Detection count
- Annotated image download

## Model

- Architecture: YOLOv8n
- Dataset: RDD2022 India
- Classes: Pothole
- Training epochs: 30
- Image size: 512 x 512

## Test Performance

| Metric | Score |
|---|---:|
| Precision | 49.29% |
| Recall | 43.21% |
| F1-score | 46.05% |
| mAP@50 | 41.86% |
| mAP@50-95 | 16.63% |

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd Pothole-Detection

pip install -r requirements.txt

streamlit run app.py
```

## Deployment

Deployed using Streamlit Community Cloud.

## Limitations

The model may miss potholes, particularly small or distant ones.
Predictions depend on image quality and the confidence threshold.
