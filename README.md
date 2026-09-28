# Face Recognition with ArcFace Embeddings

Face recognition experiment using InsightFace `buffalo_l` embeddings and cosine similarity.

## Overview

The assignment creates a representative target embedding by averaging embeddings from images in `target/`, then compares candidate faces in `celeb/` against that embedding using cosine similarity.

- **Model:** InsightFace `buffalo_l`
- **Representation:** face embedding
- **Similarity:** cosine similarity
- **Task:** retrieve candidate images judged to contain the target identity

## Reported Experiments

| Setting | Target | Reported Threshold | Accuracy@10 |
|---|---|---:|---:|
| #1 | Song Joong-ki | 0.6 | 100% |
| #2 | Han So-hee | 0.6 | 100% |
| #3 | Cha Eun-woo | 0.5 | 100% |
| #4 | Self | 0.8 | 0% |

For Setting #4, the submitted report states that nine Kim Da-mi images were selected even though the actual target identity was absent from the candidate set. The assignment evaluated this as 0% accuracy and identified threshold adjustment or additional filtering as possible improvements.

## Important Source Note

The submitted source defines `SIMILARITY_THRESHOLD = 0.8` but performs the comparison as:

```python
similarity > (1 - SIMILARITY_THRESHOLD)
```

Therefore, that code would compare against `0.2`, not `0.8`. This portfolio version fixes the comparison so the configured threshold is used directly:

```python
similarity >= threshold
```

The table above preserves the values reported in the original assignment spreadsheet; it does not claim that the corrected implementation reproduces those results without rerunning the experiment.

## Project Structure

```text
arcface-face-recognition/
├── config/
│   └── face_recognition.yaml
├── scripts/
│   ├── __init__.py
│   └── search.py
├── utils/
│   ├── __init__.py
│   └── face_recognizer.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage

Place target reference images in `data/target/` and candidate images in `data/celeb/`.

```bash
pip install -r requirements.txt
python -m scripts.search --config config/face_recognition.yaml
```

A threshold can also be supplied explicitly:

```bash
python -m scripts.search --threshold 0.6
```
