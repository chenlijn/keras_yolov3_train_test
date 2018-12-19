import sys
import argparse
from yolo import YOLO
from PIL import Image

def detect_img(yolo):
    img = '000011.jpg' # input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
        r_image.save('detected.png')
    yolo.close_session()


if __name__ == '__main__':

    # detect_img(YOLO(model_path, anchors_path, classes_path, gpu_num, image_file))
    image_file = '000011.jpg'
    detect_img(YOLO())

