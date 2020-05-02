import keras
from keras import backend as K
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
import sys
from keras.utils.generic_utils import get_custom_objects

LARGE_NUM = 1e9

def print_tensor(x, message='\n'):
    """Prints `message` and the tensor value when evaluated.
    Note that `print_tensor` returns a new tensor identical to `x`
    which should be used in the following code. Otherwise the
    print operation is not taken into account during evaluation.
    # Example
    ```python
        >>> x = K.print_tensor(x, message="x is: ")
    ```
    # Arguments
        x: Tensor to print.
        message: Message to print jointly with the tensor.
    # Returns
        The same tensor `x`, unchanged.
    """
    op = tf.print(message, x, output_stream=sys.stdout, summarize=-1)
    with tf.control_dependencies([op]):
        return tf.identity(x)

def constastive_loss(outputs, temper=0.4):

    y1, y2 = tf.split(outputs[0], 2, 0)
    bsize = K.shape(y1)[0]

    #labels = K.one_hot(K.arange(bsize), bsize*2 -1)
    labels = K.one_hot(K.arange(0, bsize), bsize)

    def mask_self(corr):
        masks =  1-K.one_hot(K.arange(bsize), bsize)
        dst = tf.boolean_mask(corr, masks)
        dst = K.reshape(dst, (bsize, bsize-1))
        return dst

    #logits_11 = K.dot(y1, K.transpose(y1)) / temper
    #logits_11 = mask_self(logits_11)
    #logits_22 = K.dot(y2, K.transpose(y2)) / temper
    #logits_22 = mask_self(logits_22)
    logits_12 = K.dot(y1, K.transpose(y2)) / temper
    #logits_12 = K.print_tensor(logits_12, 'logits_12\n')
    #logits_21 = K.dot(y2, K.transpose(y1)) / temper

    #logits_a = K.softmax(K.concatenate([logits_12, logits_11], 1))
    logits_a = K.softmax(logits_12)
    #logits_b = K.softmax(K.concatenate([logits_21, logits_22], 1))
    #logits_a = K.print_tensor(logits_a, 'logits_a\n')
    loss_a = K.categorical_crossentropy(labels, logits_a)
    #loss_b = K.categorical_crossentropy(labels, logits_b)

    loss = loss_a# + loss_b

    return loss