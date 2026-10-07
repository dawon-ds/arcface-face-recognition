# Face Recognition with ArcFace Embeddings

Face recognition experiment using InsightFace `buffalo_l` embeddings and cosine similarity.

[Portfolio](https://incredible-march-0ef.notion.site/Face-Recognition-with-ArcFace-Embeddings-3e968564df5a8159b098cbb37e81725e)

## Project Overview

The assignment creates a representative target embedding by averaging embeddings from images in `target/`, then compares candidate faces in `celeb/` against that embedding using cosine similarity.

- **Model:** InsightFace `buffalo_l`
- **Representation:** face embedding
- **Similarity:** cosine similarity
- **Task:** retrieve candidate images judged to contain the target identity

## Implementation

**Technologies:** Python, InsightFace, ONNX Runtime, OpenCV, NumPy, scikit-learn.

1. Detect faces in reference images and extract the first detected face's embedding from each image.
2. Average the reference embeddings to form the target representation.
3. Detect all faces in each candidate image.
4. Compute cosine similarity between each detected face and the target mean.
5. Copy a candidate image to the output directory if its highest face similarity meets the threshold.

Cosine similarity compares embedding directions rather than raw image pixels. This implementation uses pretrained embeddings; it does not train an ArcFace model from scratch.

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

The default configuration uses CPU inference and threshold **0.6**. Supported input extensions are `.jpg`, `.jpeg`, and `.png`. Initial execution requires the InsightFace model files to be available locally or downloadable.

Matched images are copied into `output/`, and their highest face similarity is printed. Reference embeddings are cached as `cache/<image_stem>.npy`.

A threshold can also be supplied explicitly:

```bash
python -m scripts.search --threshold 0.6
```

## Limitations & Review

- The historical Accuracy@10 values are assignment-reported results, not a benchmark of the corrected threshold implementation.
- The search script returns all threshold matches; it does not rank candidates or calculate Accuracy@10 automatically.
- Reference images use the first detected face, so multi-person reference images can introduce the wrong identity into the mean.
- Reference cache filenames use only the image stem. Different files with the same stem can collide, and modified images can reuse stale embeddings. Use unique filenames and clear the cache when references change.
- The output directory is not cleared between runs; use a fresh directory when comparing thresholds.
- A target-absent experiment is necessary to assess false matches, alongside target-present retrieval.

This project provided experience combining face detection, embedding extraction, multiple reference images, and similarity-based retrieval. The target-absent result and corrected threshold comparison highlight the importance of calibrating rejection behavior rather than judging a recognizer only on successful matches.
