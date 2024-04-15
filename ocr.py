import string

import cv2
import easyocr
import numpy as np
import torch

import RRDBNet_arch as arch

# Device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

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


class OCR:
    super_resolution_model = "asset/model/super_resolution/RRDB_ESRGAN_x4.pth"

    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)

        self.ss = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.ss.load_state_dict(torch.load(self.super_resolution_model), strict=True)
        self.ss.eval()
        self.ss = self.ss.to(device)

    def preprocess(self, image):
        # Super resolution
        # image = image * 1.0 / 255
        # image = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # image = image.unsqueeze(0)
        # image = image.to(device)
        # with torch.no_grad():
        #     output = self.ss(image).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        # image = (output * 255.0).round().astype(np.uint8)

        # Grayscale
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
        if len(text) != 7:
            return False

        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    2] in dict_char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[
                    3] in dict_char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
                (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
            return True
        else:
            return False

    @staticmethod
    def format_license(text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        license_plate_ = ''
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char,
                   6: dict_int_to_char,
                   2: dict_char_to_int, 3: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def get_text(self, image):
        image = self.preprocess(image)
        detections = self.reader.readtext(image)
        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(' ', '')

            if self.license_complies_format(text):
                return self.format_license(text), score

        return None, None
