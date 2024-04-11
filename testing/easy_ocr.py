import easyocr


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
