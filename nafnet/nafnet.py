import cv2
import numpy as np
from basicsr.models import create_model
from matplotlib import pyplot as plt

from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse


class NAFNet:
    def __init__(self):
        self.model = None

    @staticmethod
    def img2tensor(img, bgr2rgb=False, float32=True):
        img = img.astype(np.float32) / 255.
        return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

    @staticmethod
    def img2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def parse(opt_path):
        return parse(opt_path, is_train=False)

    @staticmethod
    def create_model(opt):
        return create_model(opt)

    def run(self, img):
        img = self.img2rgb(img)
        img = self.img2tensor(img)

        self.model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if self.model.opt['val'].get('grids', False):
            self.model.grids()

        self.model.test()

        if self.model.opt['val'].get('grids', False):
            self.model.grids_inverse()

        visuals = self.model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])

        return sr_img

    def set_model(self, model):
        self.model = model


if __name__ == '__main__':
    nafnet = NAFNet()

    opt_path = 'options/test/REDS/NAFNet-width64.yml'
    opt = nafnet.parse(opt_path)
    opt['dist'] = False
    model = nafnet.create_model(opt)

    nafnet.set_model(model)

    input_path = 'demo_input/blurry-reds-1.jpg'
    output_path = 'demo_output/blurry-reds-1.jpg'

    inp = cv2.imread(input_path)

    img1 = inp
    img2 = nafnet.run(inp)

    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('NAFNet output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)

    plt.show()

    cv2.waitKey(0)
