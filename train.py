from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(devices[0], 'GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

import os, signal
import glob
import argparse
import keras
from keras import backend as K
from keras.optimizers import SGD
import importlib
from keras.callbacks import Callback

import lib.clr_model
import lib.backbone.resnet
#import lib._clr_model
#import lib._backbone.resnet

import lib.loss
import lib.data_sequence
import lib.cosine_decay
import settings
import data_aug

import numpy as np
import json

class ReloadSettings(Callback):
    def __init__(self, eps = 1e-7):
        self.eps = eps

    def reload_lr(self):
        lr = K.get_value(self.model.optimizer.lr)

        if abs(lr - settings.lr) < self.eps:
            return

        K.set_value(self.model.optimizer.lr, settings.lr)
        print("lr from {} to {}".format(lr, settings.lr))

    def on_batch_end(self, batch, logs=None):
        importlib.reload(settings)

        self.reload_lr()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def parse_args():
    """
    Parse the arguments.
    """
    import datetime
    import pytz
    time_stamp = datetime.datetime.strftime(datetime.datetime.now(pytz.timezone('Japan')), '%m%d%H%M')
    default_dst = './log/train_' + time_stamp
    parser = argparse.ArgumentParser(description='Simple training script for training a SimCLR network.')
    parser.add_argument('--resume', help='resume training from _optimizer.h5 file', type=str, default='')
    parser.add_argument('--dst', help='dst dir path', type=str, default=default_dst)

    return parser.parse_args()

class TerminateOnFlag(Callback):
    def __init__(self):
        self.flag = 0

    def on_batch_end(self, batch, logs=None):
        if self.flag==1:    
            self.model.stop_training = True

class SaveModel(Callback):
    def __init__(self, model, root_dir, include_optimizer, prefix):
        self.model = model
        self.dir = root_dir
        self.include_optimizer = include_optimizer
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        fname = '{0}_model_epoch_{1:06d}.h5'.format(self.prefix, epoch)
        fname = os.path.join(self.dir, fname)
        self.model.save(fname, include_optimizer = self.include_optimizer)
        print("SaveModel {}".format(fname))

def sigint_hander(call_back):

    def handler(signum, frame):
        call_back.flag=1

    return handler

def main():
    # parse arguments
    args = parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # backbone network to CLR
    if not args.resume:

        backbone = lib.backbone.resnet.ResNet50(settings.inch)
        # append heads to backbone for training
        clr_model = lib.clr_model.get_model(backbone, settings.proj_dim)

        #backbone = lib._backbone.resnet.create_custom_resnet50(settings.inch)
        #clr_model, _ = lib._clr_model.get_model(backbone, settings.proj_dim)

    else:
        clr_model = keras.models.load_model(args.resume, compile=False)
        backbone = clr_model.get_layer('resnet50')

    # custom loss for contrastive learning
    loss = lib.loss.constastive_loss(clr_model.outputs)
    clr_model.add_loss(loss)

    clr_model.compile(
        optimizer=SGD(lr=settings.lr, momentum=0.9, nesterov=True, clipnorm=5))

    # print model summary
    clr_model.summary()

    # prepare training generator
    data_seq = lib.data_sequence.ClrImageSequence(settings.data_dirs, settings.batch_size, data_aug.data_aug, settings.num_outputs, settings.crop_size)

    log_path = os.path.join(args.dst, 'train.log')
    csv_logger = keras.callbacks.CSVLogger(log_path)
    term_ctrlc = TerminateOnFlag()
    signal.signal(signal.SIGINT, sigint_hander(term_ctrlc))
    # start training
    clr_model.fit_generator(
        generator = data_seq,
        epochs = settings.epochs,
        steps_per_epoch=len(data_seq),
        #verbose=1,
        use_multiprocessing=True,
        workers=3,
        max_queue_size=30,
        callbacks=[
            term_ctrlc,
            csv_logger,
            #ReloadSettings(),
            lib.cosine_decay.CosineAnnealingScheduler(settings.t_max, settings.lr, settings.lr_min),
            #SaveModel(clr_model, args.dst, True, "CLR"),
            #SaveModel(backbone, args.dst, False, "BACKBONE")
        ]
    )


if __name__ == '__main__':
    main()

