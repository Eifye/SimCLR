
import keras
import numpy as np
import cv2

def zscore(image):
    return image / 255.0

def preprocess_image(x, mode='tf'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    
    x = keras.backend.cast_to_floatx(x)
    if mode == 'tf':
        # x /= 127.5
        # x -= 1.
        x = zscore(x)
    elif mode == 'caffe':
        if keras.backend.image_data_format() == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] -= 103.939
                x[1, :, :] -= 116.779
                x[2, :, :] -= 123.68
            else:
                x[:, 0, :, :] -= 103.939
                x[:, 1, :, :] -= 116.779
                x[:, 2, :, :] -= 123.68
        else:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68

    return x

def rand_crop(img, sz):

    if (img.shape[0] < sz[0]) or \
        (img.shape[1] < sz[1]):
        min_side = min(img.shape[0], img.shape[1])
        max_side = max(sz)
        rate = max_side / min_side
        img = cv2.resize(img, None, fx=rate, fy=rate)
        
    if img.shape[0] == sz[0]:
        sty = 0
    else:
        sty = np.random.randint(0, img.shape[0]-sz[0])

    if img.shape[1] == sz[1]:
        stx = 0
    else:
        stx = np.random.randint(0, img.shape[1]-sz[1])

    roi = img[sty : sty + sz[0], \
                stx : stx + sz[1]].copy()

    return roi