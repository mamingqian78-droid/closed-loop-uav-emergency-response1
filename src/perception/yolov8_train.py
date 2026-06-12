from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def train_yolov8(data_dir: str | Path, model_name: str = "yolov8n-cls.pt", epochs: int = 1, imgsz: int = 96, device: str = "0"):
    model = YOLO(model_name)
    return model.train(data=str(data_dir), epochs=epochs, imgsz=imgsz, device=device, workers=0, cache=False)


if __name__ == "__main__":
    train_yolov8(Path("../data/aiderv2"))
