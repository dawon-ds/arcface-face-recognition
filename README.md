# Face Recognition with ArcFace Embeddings

[Portfolio Page](https://app.notion.com/p/3e968564df5a8159b098cbb37e81725e)

Face retrieval using pretrained InsightFace `buffalo_l` embeddings, averaging multiple target face embeddings and comparing candidate faces with cosine similarity.

## Approach

**Technologies:** Python, InsightFace, ONNX Runtime, OpenCV, NumPy, scikit-learn.

1. Detect faces in reference images and extract the first detected face's embedding from each image.
2. Average the reference embeddings to form the target representation.
3. Detect all faces in each candidate image.
4. Compute cosine similarity between each detected face and the target mean.
5. Copy a candidate image to the output directory if its highest face similarity meets the threshold.

Cosine similarity compares embedding directions rather than raw image pixels. This implementation uses pretrained embeddings; it does not train an ArcFace model from scratch.

## Results

| Setting | Target | Threshold | Accuracy@10 |
|---|---|---:|---:|
| #1 | Song Joong-ki | 0.6 | 100% |
| #2 | Han So-hee | 0.6 | 100% |
| #3 | Cha Eun-woo | 0.5 | 100% |
| #4 | Self image | 0.8 | 0% |

In the additional self-image experiment, the target identity was absent from the candidate set, but nine Kim Da-mi images were selected as matches.

## Repository Structure

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

## Limitations

The submitted source defines `SIMILARITY_THRESHOLD = 0.8` but performs the comparison as:

```python
similarity > (1 - SIMILARITY_THRESHOLD)
```

Therefore, that code would compare against `0.2`, not `0.8`. This portfolio version fixes the comparison so the configured threshold is used directly:

```python
similarity >= threshold
```

- The search script returns all matches that meet the similarity threshold rather than ranking candidates.
- Reference images use the first detected face, so multi-person reference images can introduce the wrong identity into the mean.
- Reference cache filenames use only the image stem. Different files with the same stem can collide, and modified images can reuse stale embeddings. Use unique filenames and clear the cache when references change.
- The output directory is not cleared between runs; use a fresh directory when comparing thresholds.

## Data and Third-party Resources

- **Pretrained model:** [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo), `buffalo_l`. InsightFace code is MIT licensed, but its released pretrained model packs are for **non-commercial research use** unless separately licensed.
- **Images:** reference and candidate face images are not bundled. Use only images with appropriate rights and permissions.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
