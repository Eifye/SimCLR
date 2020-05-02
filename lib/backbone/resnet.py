import keras.backend
import keras.layers
import keras.models
import keras.regularizers
from keras.utils.generic_utils import get_custom_objects

from .frn import FRN
from .block import basic_2d, bottleneck_2d

class ResNet2D(keras.Model):
    """
    custom model based on ""keras_resnet""
    batch norm -> FRN

    Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    Usage:
        >>> import keras_resnet.blocks
        >>> import keras_resnet.models
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> blocks = [2, 2, 2, 2]
        >>> block = keras_resnet.blocks.basic_2d
        >>> model = keras_resnet.models.ResNet(x, classes, blocks, block, classes=classes)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(
        self,
        inputs,
        blocks,
        block,
        include_top=True,
        classes=1000,
        numerical_names=None,
        *args,
        **kwargs
    ):

        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv1", padding="same")(inputs)
        x = FRN(name="frn_conv1")(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

        features = 64

        outputs = []

        for stage_id, iterations in enumerate(blocks):
            for block_id in range(iterations):
                x = block(
                    features,
                    stage_id,
                    block_id,
                    numerical_name=(block_id > 0 and numerical_names[stage_id]),
                )(x)

            features *= 2
            
            outputs.append(x)

        if include_top:
            assert classes > 0

            x = keras.layers.GlobalAveragePooling2D(name="pool5")(x)
            x = keras.layers.Dense(classes, activation="softmax", name="fc1000")(x)

            super(ResNet2D, self).__init__(inputs=inputs, outputs=x, *args, **kwargs)
        else:
            # Else output each stages features
            super(ResNet2D, self).__init__(inputs=inputs, outputs=outputs, *args, **kwargs)

class ResNet50(ResNet2D):
    def __init__(self, input_channels, blocks=None, include_top=False, classes=1000, *args, **kwargs):

        inputs = keras.layers.Input((None, None, input_channels))

        if blocks is None:
            blocks = [3, 4, 6, 3]

        numerical_names = [False, False, False, False]

        super(ResNet50, self).__init__(
            inputs,
            blocks,
            numerical_names=numerical_names,
            block=basic_2d,
            include_top=include_top,
            classes=classes,
            *args,
            **kwargs
        )


get_custom_objects().update({'ResNet2D': ResNet2D})
get_custom_objects().update({'ResNet50': ResNet50})