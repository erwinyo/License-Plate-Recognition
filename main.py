import cv2
from ultralytics import YOLO
import easyocr


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model, task="detect")

    def detect(self, frame, classes=None):
        return self.model.predict(frame, classes=classes, device=0)[0]

    def track(self, frame, classes=None):
        return self.model.track(frame, classes=classes, device=0, tracker="bytetrack.yaml", persist=True)[0]


class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

    @staticmethod
    def preprocess(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        image = 255 - image

        return image

    def get_text(self, image):
        image = self.preprocess(image)
        result = self.reader.readtext(image, detail=0)
        if result:
            return result[0]
        else:
            return "Unknown"


if __name__ == '__main__':
    yolo_car = YoloDetector("asset/model/yolo/yolov8l.pt")
    yolo_lpr = YoloDetector("asset/model/lpr/lpr_fast.pt")
    ocr = OCR()

    cap = cv2.VideoCapture("asset/video/sample.mp4")

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        # Resize
        frame = cv2.resize(frame, (0, 0), fx=0.70, fy=0.70)

        # Car Detection
        cars = []
        result_car = yolo_car.track(frame, classes=[2, 3, 5, 7])  # This is the index class come with YOLOv8
        for x1, y1, x2, y2, id_, score_, class_idx_ in result_car.boxes.data.tolist():
            x1, y1, x2, y2, id_, class_idx_ = int(x1), int(y1), int(x2), int(y2), int(id_), int(class_idx_)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            cars.append([x1, y1, x2, y2, id_, score_, class_idx_])

        # License Plate Detection
        result_lpr = yolo_lpr.detect(frame)
        for x1, y1, x2, y2, score_, class_idx_ in result_lpr.boxes.data:
            x1, y1, x2, y2, score_, class_idx_ = int(x1), int(y1), int(x2), int(y2), int(score_), int(class_idx_)

            for x1_car, y1_car, x2_car, y2_car, _, _, _ in cars:
                if x1 > x1_car and y1 > y1_car and x2 < x2_car and y2 < y2_car:
                    lp = frame[y1:y2, x1:x2]
                    lp_h, lp_w = lp.shape[:2]
                    frame[y1_car:y1_car + lp_h, x1_car:x1_car + lp_w] = lp

                    # OCR
                    text = ocr.get_text(lp)
                    cv2.putText(frame, text, (x1_car, y1_car), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    break

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
