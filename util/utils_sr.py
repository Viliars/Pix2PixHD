import numpy as np
import cv2
import random
from util import utils_image as util


def add_JPEG_noise(img, quality_factor):
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def get_face_pair(img_LQ, img_HQ):
    if random.random() > 0.5:
        size = int(1024//round(random.uniform(2.0, 5.0), 2))
        interpolation = random.choice([1, 2, 3])

        # resize
        img_LQ = cv2.resize(img_LQ, (size, size), interpolation=interpolation)
        img_HQ = cv2.resize(img_HQ, (size, size), interpolation=interpolation)

        interpolation = random.choice([1, 2, 3])
        # resize
        img_LQ = cv2.resize(img_LQ, (512, 512), interpolation=interpolation)
        img_HQ = cv2.resize(img_HQ, (512, 512), interpolation=interpolation)

        if random.random() > 0.5:
            quality_factor = random.randint(60, 95)

            img_LQ = add_JPEG_noise(img_LQ, quality_factor)
            img_HQ = add_JPEG_noise(img_HQ, quality_factor)

    return img_LQ, img_HQ


if __name__ == '__main__':
    import utils_image as util
    lq = util.imread_uint('/home/viliar/data/DatasetNew/train_A/00000.jpg', 3)
    hq = util.imread_uint('/home/viliar/data/DatasetNew/train_B/00000.jpg', 3)

    hq = util.uint2single(hq)
    lq = util.uint2single(lq)


    img_lq, img_hq = get_face_pair(lq, hq)
    img_lq = util.single2uint(img_lq)
    img_hq = util.single2uint(img_hq)

    img_concat = np.concatenate([img_lq, img_hq], axis=1)
    util.imsave(img_concat, 'test.png')