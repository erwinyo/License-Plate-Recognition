import cv2
import easyocr
import pytesseract
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model, task="detect")

    def detect(self, frame):
        return self.model.predict(frame, device=0, half=True, conf=0.55)[0]


class EasyOCR:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)

    def detect_text(self, frame):
        _result = self.reader.readtext(frame)
        if len(_result) == 0:
            bbox = "Unknown"
            text = "Unknown"
            confidence = "Unknown"
        else:
            bbox = _result[0][0]
            text = _result[0][1]
            confidence = _result[0][2]

        return_value = {
            "bbox": bbox,
            "text": text,
            "confidence": confidence
        }

        return return_value


if __name__ == '__main__':
    yolo = YoloDetector("asset/model/lpr/lpr.engine")
    easy_ocr = EasyOCR()

    cap = cv2.VideoCapture("asset/video/sample.mp4")

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        result = yolo.detect(frame)
        boxes = result.boxes.xyxy.clone().tolist()
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

            # Cropped only the license plate
            license_plate = frame[y1:y2, x1:x2]
            lpr = easy_ocr.detect_text(license_plate)

            cv2.putText(license_plate, f"{lpr['text']}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)

            frame[y1:y2, x1:x2] = license_plate

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
