import easyocr

if __name__ == '__main__':
    IMAGE_PATH = '../asset/image/license#1.jpg'
    reader = easyocr.Reader(['en'])
    result = reader.readtext(IMAGE_PATH)
    print(result)
    for detection in result:
        if detection[2] > 0.5:
            print(detection[1])
