import keras
from keras.utils import Sequence
from .utils.file_utils import list_image_dirs
import cv2
import numpy as np
from .utils.image import rand_crop
from .utils.stop_watch import stopwatch

class ClrImageSequence(Sequence):
    def __init__(self, data_dirs, batch_size, data_aug_callback, num_outputs, crop_size):
        self.batch_size = batch_size
        self.data_aug_callback = data_aug_callback
        self.image_paths = list_image_dirs(data_dirs)
        self.num_outputs = num_outputs
        self.crop_size = crop_size

        self._reset()

    def __getitem__(self, idx):

        start = idx*self.batch_size
        end = start+self.batch_size
        order = self.order[start:end]
        images = [self._get_image(ii) for ii in order]
        aug1 = [self._preprocess(self.data_aug_callback(img)) for img in images]
        aug2 = [self._preprocess(self.data_aug_callback(img)) for img in images]
 
        # for ii in range(self.batch_size):
        #     cv2.imwrite("dump/oo_{0:04d}.png".format(ii), images[ii])
        #     cv2.imwrite("dump/aa_{0:04d}.png".format(ii), aug1[ii]*255)
        #     cv2.imwrite("dump/bb_{0:04d}.png".format(ii), aug2[ii]*255)
        
        dst = aug1+aug2
        dst = np.array(dst, dtype=keras.backend.floatx())

        return dst, None

    def on_epoch_end(self):
        self._reset()

    def _preprocess(self, img):
        dst = img.astype(np.float32)
        dst = dst / 255.

        return dst

    def __len__(self):
        dlen = self._data_len()
        ret = dlen // self.batch_size

        return ret

    def _data_len(self):
        return len(self.image_paths)

    def _reset(self):
        dlen = self._data_len()
        self.order = np.random.permutation(dlen)

    def _erase_wb(self, image):
        _image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, non_blk = cv2.threshold(_image, 10, 255, cv2.THRESH_BINARY)
        _, non_wht = cv2.threshold(255-_image, 10, 255, cv2.THRESH_BINARY)
        target = cv2.bitwise_and(non_blk, non_wht)

        x,y,w,h = cv2.boundingRect(target)

        if w==0 or h==0:
            return image

        return image[y:y+h, x:x+w]

    #@stopwatch
    def _get_image(self, idx):
        #raw = np.fromfile(self.image_paths[idx])
        #img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            print(self.image_paths[idx])
        img = self._erase_wb(img)
        dst = rand_crop(img, self.crop_size)
        del img

        return dst