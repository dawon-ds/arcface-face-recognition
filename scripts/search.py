import argparse
import shutil
from pathlib import Path

import yaml

from utils import FaceRecognizer


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/face_recognition.yaml")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = (
        args.threshold
        if args.threshold is not None
        else cfg["recognition"]["threshold"]
    )

    target_dir = Path(cfg["paths"]["target_dir"])
    candidate_dir = Path(cfg["paths"]["candidate_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    recognizer = FaceRecognizer(
        model_name=cfg["model"]["name"],
        providers=cfg["model"]["providers"],
        cache_dir=cfg["paths"]["cache_dir"],
    )
    target_embedding = recognizer.mean_target_embedding(target_dir)

    for image_path in sorted(candidate_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        matched, similarity = recognizer.contains_target(
            image_path,
            target_embedding,
            threshold,
        )

        if matched:
            shutil.copy2(image_path, output_dir / image_path.name)
            print(
                f"match: {image_path.name} "
                f"(cosine_similarity={similarity:.4f})"
            )


if __name__ == "__main__":
    main()
