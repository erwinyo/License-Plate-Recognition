import string

import cv2
import easyocr
import torch

# Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = torch.cuda.is_available()

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

integer_string = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    @staticmethod
    def preprocess(image):
        # Grayscale + Masking
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        image = 255 - image

        return image

    @staticmethod
    def license_complies_format(text):
        """
            Check if the license plate text complies with the required format.

            Args:
                text (str): License plate text.

            Returns:
                bool: True if the license plate complies with the format, False otherwise.
        """
        length = len(text)
        if length != 6 and length != 5:
            return False, None

        if length == 6:
            letter1 = text[0] in integer_string or text[0] in dict_char_to_int.keys()
            letter2 = text[1] in integer_string or text[1] in dict_char_to_int.keys()
            letter3 = text[2] in integer_string or text[2] in dict_char_to_int.keys()
            letter4 = text[3] in integer_string or text[3] in dict_char_to_int.keys()
            letter5 = text[4] in integer_string or text[4] in dict_char_to_int.keys()
            letter6 = text[5] in integer_string or text[5] in dict_char_to_int.keys()

            if letter1 and letter2 and letter3 and letter4 and letter5 and letter6:
                return True, length

        elif length == 5:
            letter1 = text[0] in integer_string or text[0] in dict_char_to_int.keys()
            letter2 = text[1] in integer_string or text[1] in dict_char_to_int.keys()
            letter3 = text[2] in integer_string or text[2] in dict_char_to_int.keys()
            letter4 = text[3] in integer_string or text[3] in dict_char_to_int.keys()
            letter5 = text[4] in integer_string or text[4] in dict_char_to_int.keys()

            if letter1 and letter2 and letter3 and letter4 and letter5:
                return True, length

        return False, None


    @staticmethod
    def format_license_text(text, length_of_text):
        license_plate_ = ''
        mapping = {}

        if length_of_text == 6:
            mapping = {
                0: dict_char_to_int,
                1: dict_char_to_int,
                2: dict_char_to_int,
                3: dict_char_to_int,
                4: dict_char_to_int,
                5: dict_char_to_int
            }
        elif length_of_text == 5:
            mapping = {
                0: dict_char_to_int,
                1: dict_char_to_int,
                2: dict_char_to_int,
                3: dict_char_to_int,
                4: dict_char_to_int
            }

        for j in range(length_of_text):
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def get_text(self, image):
        image = self.preprocess(image)
        detections = self.reader.readtext(image, rotation_info=[90, 270], low_text=0.6, link_threshold=0.5)
        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '')

            valid, length = self.license_complies_format(text)
            if valid:
                return self.format_license_text(text, length), score

        return None, None
