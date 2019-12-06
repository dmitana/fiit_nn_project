import argparse
import os
from functools import partial
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from src.data.load_data import load_dataset
from src.data.processing import create_dataset
from src.models.models import base_model, darknet19_model, darknet19_model_2
from src.utils import timestamp


class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()


# Initialize argument parser
parser = MyArgumentParser(
    description='Train an object detection model using YOLO method.',
    fromfile_prefix_chars='@'
)
parser.add_argument(
    'train_dataset_file_path',
    type=str,
    help='File path to the train dataset file.'
)
parser.add_argument(
    '-s',
    '--img-size',
    type=int,
    nargs=2,
    default=None,
    help='Resolution to which images will be resized in the format '
         '`height width`.'
)
parser.add_argument(
    '-v',
    '--validation-dataset-file-path',
    type=str,
    default=None,
    help='File path to the validation dataset file.',
)
parser.add_argument(
    '--logdir',
    type=str,
    default=None,
    help='TensorBoard log directory.'
)
parser.add_argument(
    '--modeldir',
    type=str,
    default=None,
    help='Directory where checkpoints of models will be stored.'
)
parser.add_argument(
    '--user-lr-scheduler',
    type=bool,
    default=False,
    help='Whether to use learning rate scheduler or not (default: '
         '%(default)s).'
)

parser_hparams = parser.add_argument_group('Hyperparameters')
parser_hparams.add_argument(
    '--model-name',
    type=str,
    default='base_model',
    help='Name of the model which will be used for training. '
         'Possible values are `base_model` (default: %(default)s).'
)
parser_hparams.add_argument(
    '--grid-size',
    type=int,
    nargs=2,
    default=(16, 16),
    help='Resolution of the grid in format `grid_rows grid_cols` '
         '(default: %(default)s).',
)
parser_hparams.add_argument(
    '--batch-size',
    type=int,
    default=16,
    help='Number of samples that will be propagated through the '
         'network (default: %(default)s).',
)
parser_hparams.add_argument(
    '--epochs',
    type=int,
    default=30,
    help='Number of forward passes and backward passes of all the '
         'training examples (default: %(default)s).',
)
parser_hparams.add_argument(
    '--learning-rate',
    type=float,
    default=0.001,
    help='Determines the step size at each iteration step '
         '(default: %(default)s).',
)
parser_hparams.add_argument(
    '--bn-momentum',
    type=float,
    default=0.9,
    help='Momentum of batch normalization for the moving average'
)
parser_hparams.add_argument(
    '--lambda-coordinates',
    type=float,
    default=5.0,
    help='Lambda coordinates parameter for the YOLO loss function. '
         'Weight of the XY and WH loss (default: %(default)s).',
)
parser_hparams.add_argument(
    '--lambda-no-object',
    type=float,
    default=0.5,
    help='Lambda no object parameter for the YOLO loss function. '
         'Weight of the one part of the confidence loss '
         '(default: %(default)s).',
)


def lr_scheduler(epoch, initial_lr):
    if epoch < 50:
        return initial_lr
    else:
        return 0.00001


def train(train_xy, training_params, model_params, val_xy=None,
          model_name='base_model', img_size=None, grid_size=(16, 16),
          log_dir=None, model_dir=None, use_lr_scheduler=False):
    """
    Train an object detection model using YOLO method.

    :param train_xy: tuple, train data in the format (imgs, anns).
    :param training_params: dict, hyperparameters used for training.
        batch_size: int, number of samples that will be propagated
            through the network.
        epochs: int, number of forward passes and backward passes of
            all the training examples.
    :param model_params: dict, hyperparameters of the model.
        learning_rate: float, determines the step size at each
            iteration step.
        l_coord: float, lambda coordinates parameter for the YOLO loss
            function. Weight of the XY and WH loss.
        l_noobj: float, lambda no object parameter for the YOLO loss
            function. Weight of the one part of the confidence loss.
    :param val_xy: tuple (default: None), validation data in the format
        (imgs, anns).
    :param model_name: str (default: base_model), name of the model to
        be used for training. Possible values are `base_model`.
    :param img_size: tuple (default: None), new resolution
        (new_img_height, new_img_width) of each image. If `None` then
        images will not be resized.
    :param grid_size: tuple (default: (16, 16)), number of
        (grid_rows, grid_cols) of grid cell.
    :param log_dir: str (default: None), TensorBoard log directory.
    :param model_dir: str (default: None), Directory where checkpoints
        of models will be stored.
    :param use_lr_scheduler: bool (default: False), whether to use
        learning rate scheduler or not.
    :return:
        model: tf.keras.Model, trained model.
        history: History, its History.history attribute is a record of
            training loss values and metrics values at successive
            epochs, as well as validation loss values and validation
            metrics values (if applicable).
    """
    # Create train dataset
    train_dataset = create_dataset(
        train_xy[0],
        train_xy[1],
        img_size,
        grid_size,
        is_training=True,
        batch_size=training_params['batch_size']
    )

    # Create validation dataset
    val_dataset = None
    if val_xy is not None:
        val_dataset = create_dataset(
            val_xy[0],
            val_xy[1],
            img_size,
            grid_size,
            is_training=False,
            batch_size=training_params['batch_size']
        )

    # Choose model
    input_shape = train_xy[0].shape[1:]
    if model_name == 'base_model':
        model = base_model(grid_size, input_shape=input_shape, **model_params)
    elif model_name == 'darknet19_model':
        model = darknet19_model(
            grid_size,
            input_shape=input_shape,
            **model_params
        )
    elif model_name == 'darknet19_model_2':
        model = darknet19_model_2(grid_size, input_shape=input_shape, **model_params)
    else:
        raise ValueError(f'Error: undefined model `{model_name}`.')

    # Create keras callbacks
    callbacks = []
    if log_dir is not None:
        log_dir = os.path.join(log_dir, timestamp())
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                profile_batch=0,
            )
        )
    if model_dir is not None:
        model_path = os.path.join(model_dir, timestamp(), 'model.ckpt')
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True,
                mode='min',
                verbose=1
            )
        )
    if use_lr_scheduler:
        fn_scheduler = partial(
            lr_scheduler,
            initial_lr=model_params['learning_rate']
        )
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
            fn_scheduler
        ))

    # Train model
    model.summary()
    history = model.fit(
        train_dataset,
        epochs=training_params['epochs'],
        validation_data=val_dataset,
        steps_per_epoch=len(train_xy[1]) // training_params['batch_size'],
        callbacks=callbacks,
    )

    # TensorBoard HParams saving
    if log_dir is not None:
        log_dir_hparams = os.path.join(log_dir, 'hparams')
        with tf.summary.create_file_writer(log_dir_hparams).as_default():
            hp.hparams({**training_params, **model_params}, trial_id=log_dir)

            train_best_loss = min(history.history['loss'])
            train_best_f1_score = max(history.history['F1Score'])
            tf.summary.scalar('train_best_loss', train_best_loss, step=0)
            tf.summary.scalar(
                'train_best_f1_score',
                train_best_f1_score,
                step=0
            )

            if val_dataset is not None:
                val_best_loss = min(history.history['val_loss'])
                val_best_f1_score = max(history.history['val_F1Score'])
                tf.summary.scalar('val_best_loss', val_best_loss, step=0)
                tf.summary.scalar(
                    'val_best_f1_score',
                    val_best_f1_score,
                    step=0
                )

    return model, history


if __name__ == '__main__':
    args = parser.parse_args()

    print('Load train data')
    train_data = load_dataset(args.train_dataset_file_path)
    print('Load validation data')
    val_data = None if args.validation_dataset_file_path is None \
        else load_dataset(args.validation_dataset_file_path)

    train(
        train_xy=train_data,
        training_params={
            'batch_size': args.batch_size,
            'epochs': args.epochs
        },
        model_params={
            'learning_rate': args.learning_rate,
            'bn_momentum': args.bn_momentum,
            'l_coord': args.lambda_coordinates,
            'l_noobj': args.lambda_no_object,
        },
        val_xy=val_data,
        model_name=args.model_name,
        img_size=args.img_size,
        grid_size=args.grid_size,
        log_dir=args.logdir,
        model_dir=args.modeldir,
        use_lr_scheduler=args.use_lr_scheduler,
    )
