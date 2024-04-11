import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model, task="detect")

    def detect(self, frame, classes=None):
        return self.model.track(frame, classes=classes, device=0)[0]


if __name__ == '__main__':
    yolo_car = YoloDetector("asset/model/yolo/yolov8l.onnx")
    yolo_lpr = YoloDetector("asset/model/lpr/lpr.onnx")

    cap = cv2.VideoCapture("asset/video/sample.mp4")

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        # License Plate Detection
        # result_lp = yolo_lpr.detect(frame)
        # boxes_lp = result_lp.boxes.xyxy.clone().tolist()

        # Car Detection
        result_car = yolo_car.detect(frame, classes=[2, 3, 5, 7])
        boxes_car = result_car.boxes.xyxy.clone().tolist()
        ids_car = result_car.boxes.id.clone().tolist()

        for id_car, box_car in zip(ids_car, boxes_car):
            x1, y1, x2, y2 = box_car
            x1, y1, x2, y2, id_car = int(x1), int(y1), int(x2), int(y2), int(id_car)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(frame, f"{id_car}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
