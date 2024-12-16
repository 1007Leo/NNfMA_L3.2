import os
import json
import bentoml
from PIL.Image import Image
from ultralytics import YOLO

@bentoml.service(resources={"gpu": 1})
class YoloV8:
    def __init__(self):
        model = os.getenv("YOLO_MODEL", "./best.pt")
        self.model = YOLO(model)

    @bentoml.api()
    def predict(self, images: Image) -> list[dict]:
        results = self.model.predict(images)[0]
        return json.loads(results.to_json())
