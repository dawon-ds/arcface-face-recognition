from pathlib import Path

import cv2
import insightface
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognizer:
    def __init__(self, model_name="buffalo_l", providers=None, cache_dir="cache"):
        providers = providers or ["CPUExecutionProvider"]
        self.model = insightface.app.FaceAnalysis(
            name=model_name,
            providers=providers,
        )
        self.model.prepare(ctx_id=0)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_image(path):
        path = Path(path)
        return cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

    def embedding(self, image_path, cache=True):
        image_path = Path(image_path)
        cache_path = self.cache_dir / f"{image_path.stem}.npy"

        if cache and cache_path.exists():
            return np.load(cache_path)

        image = self._read_image(image_path)
        if image is None:
            return None

        faces = self.model.get(image)
        if not faces:
            return None

        emb = faces[0].embedding
        if cache:
            np.save(cache_path, emb)
        return emb

    def mean_target_embedding(self, target_dir):
        embeddings = []
        for path in Path(target_dir).iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            emb = self.embedding(path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            raise RuntimeError("No target face embeddings could be created.")

        return np.mean(embeddings, axis=0, keepdims=True)

    def contains_target(self, image_path, target_embedding, threshold):
        image = self._read_image(image_path)
        if image is None:
            return False, None

        faces = self.model.get(image)
        if not faces:
            return False, None

        best_similarity = None
        for face in faces:
            emb = face.embedding.reshape(1, -1)
            similarity = float(
                cosine_similarity(emb, target_embedding)[0][0]
            )
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity

        return best_similarity >= threshold, best_similarity
