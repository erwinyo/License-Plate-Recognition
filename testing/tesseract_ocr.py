from pytesseract import *
import cv2


if __name__ == '__main__':
    image = cv2.imread("../asset/image/license#1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("Threshold", image)
    image = 255 - image
    cv2.imshow("Inverted", image)

    result = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
    print(result)

    cv2.waitKey(0)


