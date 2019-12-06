import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, \
    Reshape, Input, LeakyReLU, BatchNormalization, Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam, SGD
from src.models.losses import YoloLoss
from src.models.metrics import F1Score
import tensorflow as tf


def base_model(grid_size, input_shape, n_categories=0,
               **hparams):
    """
    Base model for object detection using the YOLO method.

    Base is adopted from
    https://github.com/ecaradec/humble-yolo/blob/master/main.py
    and modified.

    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
    :param input_shape: tuple, shape of input images, without batch
        dimension.
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

    # Layer 1
    x = Conv2D(
        filters=16,
        kernel_size=1,
        activation='relu',
        padding='same',
        name='conv_1',
    )(inputs)

    # Layer 2
    x = Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv_2',
    )(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pooling_1',
    )(x)

    # Layer 3
    x = Conv2D(
        filters=16,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv_3',
    )(x)

    # Layer 4
    x = Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        padding='same',
        name='conv_4',
    )(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pooling_2',
    )(x)

    # Layer 5
    x = Flatten(name='flatten')(x)
    x = Dense(
        units=256,
        activation='sigmoid',
        name='dense_1',
    )(x)

    # Layer 6
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
        metrics=[F1Score(iou_threshold=0.5, grid_size=grid_size)]
    )

    return model


def darknet19_model(grid_size, input_shape, n_categories=0,
                    **hparams):
    """
    Darknet19 model for object detection using the YOLO method.

    Darknet19 model is adopted from YOLO 9000 (v2).

    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
    :param input_shape: tuple, shape of input images, without batch
        dimension.
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
    momentum = hparams['bn_momentum']
    act = None

    inputs = Input(input_shape)

    # Layer 1
    x = Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_1',
    )(inputs)
    x = BatchNormalization(momentum=momentum, name='bn_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pool_1',
    )(x)

    # Layer 2
    x = Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_2',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pool_2',
    )(x)

    # Layer 3
    x = Conv2D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_3',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation=act,
        name='conv_4',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_5',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pool_3',
    )(x)

    # Layer 6
    x = Conv2D(
        filters=256,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_6',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(
        filters=128,
        kernel_size=1,
        padding='same',
        activation=act,
        name='conv_7',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(
        filters=256,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_8',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pool_4',
    )(x)

    # Layer 9
    x = Conv2D(
        filters=512,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_9',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        activation=act,
        name='conv_10',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(
        filters=512,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_11',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        activation=act,
        name='conv_12',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(
        filters=512,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_13',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_13')(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip_connection = x
    x = MaxPooling2D(
        pool_size=(2, 2),
        name='max_pool_5',
    )(x)

    # Layer 14
    x = Conv2D(
        filters=1024,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_14',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(
        filters=512,
        kernel_size=1,
        padding='same',
        activation=act,
        name='conv_15',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(
        filters=1024,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_16',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(
        filters=512,
        kernel_size=1,
        padding='same',
        activation=act,
        name='conv_17',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(
        filters=1024,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_18',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(
        filters=1024,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_19',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(
        filters=1024,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_20',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    x = Conv2D(
        filters=1024,
        kernel_size=3,
        padding='same',
        activation=act,
        name='conv_21',
    )(x)
    x = BatchNormalization(momentum=momentum, name='bn_21')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = tf.nn.space_to_depth(skip_connection, 2)
    x = concatenate([x, skip_connection])

    # Layer 22
    outputs = Conv2D(
        filters=5 + n_categories,
        kernel_size=1,
        padding='same',
        activation='sigmoid',
        name='conv_22',
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='darknet19_model',
    )
    model.compile(
        optimizer=Adam(learning_rate=hparams['learning_rate']),
        loss=YoloLoss(
            grid_size,
            l_coord=hparams['l_coord'],
            l_noobj=hparams['l_noobj']
        ),
        metrics=[F1Score(iou_threshold=0.5, grid_size=grid_size)]
    )

    return model


def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)


def darknet19_model_2(grid_size, input_shape=(416, 416, 3), n_categories=0,
                    **hparams):
    """
    Darknet19 model for object detection using the YOLO method.

    Darknet19 model is adopted from YOLO 9000 (v2).

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
    act = None

    inputs = Input(input_shape)

    input_image = Input(shape=input_shape)

    # Layer 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1',
               use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                             name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22',
               use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    output = Conv2D((4 + 1), (1, 1), strides=(1, 1), padding='same',
               name='conv_23', activation='sigmoid')(x)

    model = keras.Model(
        inputs=input_image,
        outputs=output,
        name='darknet19_model',
    )
    model.compile(
        optimizer=Adam(learning_rate=hparams['learning_rate']),
        loss=YoloLoss(
            grid_size,
            l_coord=hparams['l_coord'],
            l_noobj=hparams['l_noobj']
        ),
        metrics=[F1Score(iou_threshold=0.5, grid_size=grid_size)]
    )

    return model
