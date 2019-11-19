import argparse
from src.data.load_data import load_dataset
from src.data.processing import create_dataset
from src.models.models import base_model

# Initialize argument parser
parser = argparse.ArgumentParser(
    description='Train an object detection model using YOLO method.',
    fromfile_prefix_chars='@'
)
parser.add_argument(
    'train_dataset_file_path',
    type=str,
    help='File path to the train dataset file.'
)
parser.add_argument(
    'img_size',
    type=int,
    nargs=2,
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


def train(train_xy, val_xy, model_name, img_size, grid_size, training_params,
          model_params):
    """
    Train an object detection model using YOLO method.

    :param train_xy: tuple, train data in the format (imgs, anns).
    :param val_xy: tuple, validation data in the format (imgs, anns).
    :param model_name: str, name of the model to be used for training.
        Possible values are `base_model`.
    :param img_size: tuple, (img_height, img_width) of each image.
    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
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
    :return:
        model: tf.keras.Model, trained model.
        history: History, its History.history attribute is a record of
            training loss values and metrics values at successive
            epochs, as well as validation loss values and validation
            metrics values (if applicable).
    """
    train_dataset = create_dataset(
        train_xy[0],
        train_xy[1],
        img_size,
        grid_size,
        is_training=True,
        batch_size=training_params['batch_size']
    )

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

    if model_name == 'base_model':
        model = base_model(grid_size, **model_params)
    else:
        raise ValueError(f'Error: undefined model `{model_name}`.')

    model.summary()
    history = model.fit(
        train_dataset,
        epochs=training_params['epochs'],
        validation_data=val_dataset,
        steps_per_epoch=len(train_xy[1]) // training_params['batch_size']
    )

    return model, history


if __name__ == '__main__':
    args = parser.parse_args()

    train_data = load_dataset(args.train_dataset_file_path)
    val_data = None if args.validation_dataset_file_path is None \
        else load_dataset(args.val_dataset_file_path)

    train(
        train_data,
        val_data,
        args.model_name,
        args.img_size,
        args.grid_size,
        training_params={
            'batch_size': args.batch_size,
            'epochs': args.epochs
        },
        model_params={
            'learning_rate': args.learning_rate,
            'l_coord': args.lambda_coordinates,
            'l_noobj': args.lambda_no_object,
        }
    )
