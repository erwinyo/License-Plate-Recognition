from ultralytics import YOLO

class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model, task="detect")

    def detect(self, frame, classes=None, conf=0.25):
        return self.model.predict(frame, classes=classes, conf=conf, device=0)[0]

    def track(self, frame, classes=None, conf=0.25):
        return self.model.track(frame, classes=classes, conf=conf, device=0, tracker="bytetrack.yaml", persist=True)[0]
