import argparse
from src.data.load_data import load_dataset
from src.data.processing import create_dataset
from src.models.models import base_model

# Initialize argument parser
parser = argparse.ArgumentParser(
    description='',
    fromfile_prefix_chars='@'
)
parser.add_argument(
    'train_dataset_file_path',
    type=str,
    help=''
)
parser.add_argument(
    'img_size',
    type=int,
    nargs=2,
    help=''
)
parser.add_argument(
    '-v',
    '--validation-dataset-file-path',
    type=str,
    default=None,
    help='',
)

parser_hparams = parser.add_argument_group('Hyperparameters')
parser_hparams.add_argument(
    '--model-name',
    type=str,
    default='base_model',
    help=''
)
parser_hparams.add_argument(
    '--grid-size',
    type=int,
    nargs=2,
    default=(16, 16),
    help='',
)
parser_hparams.add_argument(
    '--batch-size',
    type=int,
    default=16,
    help='',
)
parser_hparams.add_argument(
    '--epochs',
    type=int,
    default=30,
    help='',
)
parser_hparams.add_argument(
    '--learning-rate',
    type=float,
    default=0.001,
    help='',
)
parser_hparams.add_argument(
    '--lambda-coordinates',
    type=float,
    default=5.0,
    help='',
)
parser_hparams.add_argument(
    '--lambda-no-object',
    type=float,
    default=0.5,
    help='',
)


def train(train_xy, val_xy, model_name, img_size, grid_size, training_params,
          model_params):
    """

    :param train_xy:
    :param val_xy:
    :param model_name:
    :param img_size:
    :param grid_size:
    :param training_params:
    :param model_params:
    :return:
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
    model.fit(
        train_dataset,
        epochs=training_params['epochs'],
        validation_data=val_dataset,
        steps_per_epoch=len(train_xy[1]) // training_params['batch_size']
    )


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
