from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K


class YoloLoss(Loss):
    """
    Custom loss function for object detection task using the YOLO
    method.

    Loss function is composed of four loss functions:
        1) XY loss. This loss compute error of (x, y) bounding box
            coordinates.
        2) WH loss. This loss compute error of bounding box width and
            height.
        3) Confidence loss. This loss compute error of bounding box
            confidence. Intersection over Union (IoU) is used.
        4) Classification loss. This loss compute error of category
            classifier. CURRENTLY THIS LOSS IS NOT IMPLEMENTED YET.
    """
    def __init__(self, grid_size, l_coord=5.0, l_noobj=0.5):
        """
        Construct new object of `YoloLoss` class.

        :param grid_size: tuple, number of (grid_rows, grid_cols) of
            grid cell.
        :param l_coord: float (default: 5.0), lambda coordinates
            parameter. Weight of the XY and WH loss.
        :param l_noobj: float (default: 0.5), lambda no object
            parameter. Weight of the one part of the confidence loss.
        """
        super(YoloLoss, self).__init__(name='yolo_loss')

        self.grid_rows, self.grid_cols = grid_size
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def call(self, y_true, y_pred):
        """
        Computation of the YOLO loss.

        Loss computation is adopted from
        https://github.com/ecaradec/humble-yolo/blob/master/main.py
        and modified.

        :param y_true: Tensor dim=(batch, grid_rows, grid_cols,
            5 + n_categories), ground truth labels for each sample in
            the batch.
        :param y_pred: Tensor dim=(batch, grid_rows, grid_cols,
            5 + n_categories), predicted labels for each sample in the
            batch.
        :return: Tensor dim=(), total YOLO loss (xy loss + wh loss +
            confidence loss + classification loss).
        """
        # Extract needed  values
        pred_boxes = K.reshape(
            x=y_pred[..., :5],
            shape=(-1, self.grid_rows * self.grid_cols, 5)
        )
        true_boxes = K.reshape(
            x=y_true[..., :5],
            shape=(-1, self.grid_rows * self.grid_cols, 5)
        )

        y_true_conf = true_boxes[..., 0]
        y_true_xy = true_boxes[..., 1:3]
        y_true_wh = true_boxes[..., 3:]

        y_pred_conf = pred_boxes[..., 0]
        y_pred_xy = pred_boxes[..., 1:3]
        y_pred_wh = pred_boxes[..., 3:]

        # XY loss
        xy_loss = self.l_coord * K.sum(
            K.sum(
                K.square(y_pred_xy - y_true_xy),
                axis=-1
            )
            * y_true_conf,
            axis=-1
        )

        # WH loss
        wh_loss = self.l_coord * K.sum(
            K.sum(
                K.square(K.sqrt(y_pred_wh) - K.sqrt(y_true_wh)),
                axis=-1
            )
            * y_true_conf,
            axis=-1
        )

        # TODO: confidence loss looks bugged, test it and fix it
        # Confidence (IoU) loss
        intersect_wh = K.maximum(
            K.zeros_like(y_pred_wh),
            (y_pred_wh + y_true_wh) / 2 - K.abs(y_pred_xy - y_true_xy)
        )
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        true_area = y_true_wh[..., 0] * y_true_wh[..., 1]
        pred_area = y_pred_wh[..., 0] * y_pred_wh[..., 1]
        union_area = pred_area + true_area - intersect_area
        iou = intersect_area / union_area

        conf_loss1 = K.sum(K.sum(K.square(y_pred_conf - iou), axis=-1) * y_true_conf, axis=-1)
        conf_loss2 = 0.5 * K.sum(K.sum(K.square(y_pred_conf - iou), axis=-1) * (1 - y_true_conf), axis=-1)
        conf_loss = conf_loss1 + conf_loss2
        # conf_loss = K.sum(
        #     K.square(y_true_conf * iou - y_pred_conf) * y_true_conf,
        #     axis=-1
        # )

        return xy_loss + wh_loss + conf_loss
