import cv2
import numpy as np
import settings
import keras
import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from lib.utils.stop_watch import stopwatch

def truncated_norm(lower, upper, mu, sigma, to_int=False):
    value = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs()
    return int(round(value)) if to_int else value

def _rand_rotate(image):
    size = image.shape[:2]
    rad = np.deg2rad(truncated_norm(-1.5, 1.5, 0, 0.5))
    scale = truncated_norm(0.9, 1.1, 1, 0.01)
    
    affine_matrix = np.float32([
        [np.cos(rad), -1 * np.sin(rad), 0],
        [np.sin(rad), np.cos(rad), 0]
    ])

    affine_matrix *= scale

    return cv2.warpAffine(image, affine_matrix, (size[1],size[0]), flags=cv2.INTER_LINEAR)

def _crop(img):

    if (img.shape[0] < settings.crop_size[0]) or \
        (img.shape[1] < settings.crop_size[1]):
        min_side = min(img.shape[0], img.shape[1])
        max_side = max(settings.crop_size)
        rate = max_side / min_side
        img = cv2.resize(img, None, fx=rate, fy=rate)

    sty = np.random.randint(0, img.shape[0]-settings.crop_size[0])
    stx = np.random.randint(0, img.shape[1]-settings.crop_size[1])

    roi = img[sty : sty + settings.crop_size[0], \
                stx : stx + settings.crop_size[1]].copy()

    return roi

def _rand_filp_horizontal(img):

    if np.random.uniform() < 0.5:
        return img

    return img[:, ::-1]

def _rand_brightness(img, strength):

    delta = np.random.uniform(-strength, strength)
    dst = img + delta
    dst = np.clip(dst, 0, 1)

    return dst

def _rand_contrast(img, strength):

    scale = np.random.uniform(1-strength, 1+strength)
    imean = img.mean(axis=(0,1), keepdims=True)
    dst = (img-imean)*scale + imean
    dst = np.clip(dst, 0, 1)

    return dst

def _rand_hue(img, strength):

    delta = np.random.uniform(-strength*360, strength*360) # strength_rate to degree range. opencv hue is [0..360]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] += delta

    # opencv handles <0 and 360< value
    dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return dst

def _rand_saturate(img, strength):

    delta = np.random.uniform(-strength, strength) # strength_rate to degree range. opencv hue is [0..360]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] += delta
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 1)

    # opencv handles <0 and 360< value
    dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return dst

def _blur(image):
    return cv2.blur(image, (3,3))

#@stopwatch
def data_aug(img):
    op = [_rand_contrast, _rand_hue, _rand_saturate]
    op_atrand = [_rand_rotate, _rand_rotate, _rand_filp_horizontal, _blur]
    prob_atrand = [0.2, 0.2, 0.2, 0.2]
    order = np.arange(len(op))
    np.random.shuffle(order)

    dst = img 
    dst = dst.astype(np.float32) / 255.

    for ii in order:
        dst = op[ii](dst, settings.strength)
        dst = np.clip(dst, 0, 1)

    for ii in range(len(op_atrand)):
        pp = np.random.uniform()

        if pp < prob_atrand[ii]:
            dst = op_atrand[ii](dst)
            
    dst = (dst * 255).astype(np.uint8)

    return dst