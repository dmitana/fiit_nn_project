import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow import reshape, map_fn
import tensorflow


class MeanAveragePrecision(Metric):
    """
    Mean average precision for evaluating YOLO accuracy.
    """
    def __init__(self, iou_threshold, grid_size, **kwargs):
        super(MeanAveragePrecision, self).__init__(name="MeanAveragePrecision",
                                                   **kwargs)
        self.iou_threshold = iou_threshold
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.last_recall = self.add_weight(name='recall', initializer='zeros')
        self.auc = self.add_weight(name='auc', initializer='zeros')

        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    # TODO: might be good to pick outside to function because calculated
    #  in loss
    def _calculate_intersect_over_union(self, y_true, y_pred):
        pred_boxes = K.reshape(
            x=y_pred[..., :5],
            shape=(-1, self.grid_rows * self.grid_cols, 5)
        )
        true_boxes = K.reshape(
            x=y_true[..., :5],
            shape=(-1, self.grid_rows * self.grid_cols, 5)
        )

        y_true_xy = true_boxes[..., 1:3]
        y_true_wh = true_boxes[..., 3:]

        y_pred_xy = pred_boxes[..., 1:3]
        y_pred_wh = pred_boxes[..., 3:]

        intersect_wh = K.maximum(
            K.zeros_like(y_pred_wh),
            (y_pred_wh + y_true_wh) / 2 - K.abs(y_pred_xy - y_true_xy)
        )
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        true_area = y_true_wh[..., 0] * y_true_wh[..., 1]
        pred_area = y_pred_wh[..., 0] * y_pred_wh[..., 1]
        union_area = pred_area + true_area - intersect_area
        return intersect_area / union_area

    def _classify_prediction(self, iou, true_confidence):
        if iou >= self.iou_threshold:
            if true_confidence == 1.0:
                self.true_positives.assign_add(1)
            else:
                self.false_negatives.assign_add(1)
        else:
            if true_confidence == 1.0:
                self.false_positives.assign_add(1)
        return 1.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_iou = self._calculate_intersect_over_union(y_true, y_pred)

        true_boxes = K.reshape(
            x=y_true[..., :5],
            shape=(-1, self.grid_rows * self.grid_cols, 5)
        )

        reshaped_batch_iou = reshape(batch_iou, shape=[-1])
        reshaped_true_conf = reshape(true_boxes[..., 0], shape=[-1])
        map_fn(lambda x: self._classify_prediction(x[0], x[1]),
               [reshaped_batch_iou, reshaped_true_conf],
               dtype=tensorflow.float32)

        new_precision = self.true_positives / (self.true_positives +
                                               self.false_positives)
        new_recall = self.true_positives / (self.true_positives +
                                            self.false_negatives)

        if new_precision < self.precision:
            self.auc.assign_add(
                self.precision * (new_recall - self.last_recall)
            )
            self.last_recall.assign(new_recall)
        self.precision.assign(new_precision)

    # TODO: check when result and update_state are called
    def result(self):
        return self.auc
