import keras
import keras.backend as K
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

from .backbone.resnet import ResNet50

def _split(xx, num, axis):
    return keras.layers.Lambda(lambda x: tf.split(xx, num_or_size_splits=num, axis=axis))(xx)

def _head_block(hidden, scale, proj_dim = 128):

    _hiddens = [hidden]

    if scale != 1:
        
        vsplits = _split(hidden, scale, axis=1)
        _hiddens = [_split(vsp, scale, axis=2) for vsp in vsplits]

def _l2_normalize(vec, eps=1e-6):
    alpha = keras.backend.pow(vec, 2)
    alpha = keras.backend.sum(alpha, axis=1, keepdims=True)
    alpha = keras.backend.sqrt(alpha)
    alpha = 1. / (alpha+eps)

    return vec * alpha

def _projection_head(hiddens, proj_dim=128):
    outs = []

    for hidden in hiddens:
        gap = keras.layers.GlobalMaxPooling2D()(hidden)
        h0 = keras.layers.Dense(proj_dim)(gap)
        h0 = keras.layers.Activation("relu")(h0)
        h1 = keras.layers.Dense(proj_dim)(h0)
        #h1 = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(h1)
        h1 = keras.layers.Lambda(lambda xx: _l2_normalize(xx))(h1)
        outs.append(h1)

    return outs

def get_model(backbone_model, proj_dim=128, input_channnels=3):

    inputs = keras.layers.Input((None, None, input_channnels))

    hiddens = backbone_model(inputs)
    
    outs = _projection_head([hiddens[-1]], proj_dim)

    return keras.models.Model(inputs=inputs, outputs=outs, name="sim_clr")

get_custom_objects().update({'_l2_normalize': _l2_normalize})