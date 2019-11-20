import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, \
    Reshape, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from src.models.losses import YoloLoss
from src.models.metrics import MeanAveragePrecision


def base_model(grid_size, input_shape=(256, 256, 3), n_categories=0,
               **hparams):
    """
    Base model for object detection using the YOLO method.

    Base is adopted from
    https://github.com/ecaradec/humble-yolo/blob/master/main.py
    and modified.

    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
    :param input_shape: tuple (default: (256, 256, 3), shape of input
        images, without batch dimension.
    :param n_categories: int (default: 0), number of categories to be
        detected.
    :param hparams: dict, training hyperparameters.
        learning_rate: float (default: None), determines the step size
            at each iteration step.
        l_coord: float (default: 5.0), lambda coordinates parameter
            for YOLO loss function. Weight of the XY and WH loss.
        l_noobj: float (default: 0.5), lambda no object parameter for
            YOLO loss function. Weight of the one part of the
            confidence loss.
    :return: tf.keras.Model, compiled model.
    """
    (grid_rows, grid_cols) = grid_size

    inputs = Input(input_shape)
    x = Conv2D(
        filters=16,
        kernel_size=1,
        activation='relu',
        name='conv_1',
    )(inputs)
    x = Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        name='conv_2',
    )(x)
    x = LeakyReLU(
        alpha=0.3,
        name='leaky_relu_1',
    )(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pooling_1',
    )(x)
    x = Conv2D(
        filters=16,
        kernel_size=3,
        activation='relu',
        name='conv_3',
    )(x)
    x = Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        name='conv_4',
    )(x)
    x = LeakyReLU(
        alpha=0.3,
        name='leaky_relu_2',
    )(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pooling_2',
    )(x)
    x = Flatten(name='flatten')(x)
    x = Dense(
        units=256,
        activation='sigmoid',
        name='dense_1',
    )(x)
    x = Dense(
        units=grid_rows * grid_cols * (5 + n_categories),
        activation='sigmoid',
        name='dense_2',
    )(x)
    outputs = Reshape(
        target_shape=(grid_rows, grid_cols, (5 + n_categories)),
        name='reshape_1',
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='base_model',
    )
    model.compile(
        optimizer=Adam(learning_rate=hparams['learning_rate']),
        loss=YoloLoss(
            grid_size,
            l_coord=hparams['l_coord'],
            l_noobj=hparams['l_noobj']
        ),
        metrics=[MeanAveragePrecision(iou_threshold=0.5, grid_size=grid_size)]
    )

    return model
