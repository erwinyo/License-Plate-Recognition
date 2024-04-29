from collections import deque

import cv2

from ocr import OCR
from yolo_detector import YoloDetector
from nafnet import NAFNet


def main():
    # Setup all the models
    yolo_car = YoloDetector("asset/model/yolo/yolov8l.onnx")
    yolo_lpr = YoloDetector("asset/model/lpr/lpr_fast.onnx")
    ocr = OCR()
    nafnet = NAFNet()

    # Set up the NAFNet
    opt_path = 'nafnet/options/test/REDS/NAFNet-width64.yml'
    opt = nafnet.parse(opt_path)
    opt['dist'] = False
    model = nafnet.create_model(opt)
    nafnet.set_model(model=model)

    cap = cv2.VideoCapture("asset/video/sample#1.mp4")
    result = {}
    """
        variable "result", basically will have structure like this
        result = {
            1: {
                "text": deque(maxlen=20),
                "score": deque(maxlen=20)
            },
            ...
        }
    """

    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.90, fy=0.90)  # Resize 90% height x 90% width

        # Car Detection
        result_car = yolo_car.track(frame, classes=[2, 3, 5, 7])  # This is the index class come with YOLOv8
        ids_car = [int(i) for i in result_car.boxes.id.tolist()]

        # Remove old element from dictionary for old car id
        for key in list(result.keys()):
            if key not in ids_car:
                del result[key]

        # Adding new element to dictionary for new car id
        for id_ in ids_car:
            if id_ not in result:
                result[id_] = {
                    "text": deque(maxlen=20),
                    "score": deque(maxlen=20)
                }

        # Car detection ... 1
        cars = []
        for x1, y1, x2, y2, id_, score_, _ in result_car.boxes.data.tolist():
            x1, y1, x2, y2, id_, = int(x1), int(y1), int(x2), int(y2), int(id_)
            cars.append([x1, y1, x2, y2, id_])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)  # Showing the car bounding box

        # License plate detection ... 2
        result_lpr = yolo_lpr.detect(frame, conf=0.4)
        for x1, y1, x2, y2, score_, _ in result_lpr.boxes.data:
            x1, y1, x2, y2, score_ = int(x1), int(y1), int(x2), int(y2), int(score_)

            # Looping through cars, check whether the license plate bounding box inside of that car
            for x1_car, y1_car, x2_car, y2_car, id_car in cars:
                if x1 > x1_car and y1 > y1_car and x2 < x2_car and y2 < y2_car:
                    # Cropped the license plate only
                    lp = frame[y1:y2, x1:x2]
                    lp_h, lp_w = lp.shape[:2]
                    frame[y1_car:y1_car + lp_h, x1_car:x1_car + lp_w] = lp  # showing the license plate

                    # Deblur
                    deblur = nafnet.run(lp)
                    shifty1 = y1_car + lp_h
                    frame[shifty1:shifty1 + lp_h, x1_car:x1_car + lp_w] = deblur  # Showing the deblur license plate

                    # OCR
                    text, score = ocr.get_text(lp)
                    if text is not None and score is not None:
                        result[id_car]["text"].append(text)
                        result[id_car]["score"].append(score)
                        break

        # Filter the best prediction from deque
        for x1_car, y1_car, x2_car, y2_car, id_car in cars:
            t = result[id_car]["text"]
            s = result[id_car]["score"]
            if t:
                max_score_index = s.index(max(s))
                text = t[max_score_index]

                cv2.putText(frame, text, (x1_car, y1_car), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)  # Showing license plate text

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
