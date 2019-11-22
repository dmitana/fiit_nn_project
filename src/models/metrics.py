import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow import reshape, map_fn
from tensorflow.compat.v1 import Session
import tensorflow as tf
import sys
import numpy as np


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
        self.precisions = []
        self.recalls = []

        self.last_precision = self.add_weight(name='precision',
                                              initializer='zeros')
        self.last_recall = self.add_weight(name='recall', initializer='zeros')
        self.last_used_recall = self.add_weight(name='used_recall',
                                                initializer='zeros')
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

    def _classify_prediction(self, iou, true_confidence, max_positives):
        if iou >= self.iou_threshold:
            if true_confidence == 1.0:
                self.true_positives.assign_add(1)
            else:
                self.false_positives.assign_add(1)
        else:
            if true_confidence == 1.0:
                self.false_negatives.assign_add(1)

        new_precision = self.true_positives / (self.true_positives +
                                               self.false_positives)
        new_recall = self.true_positives / max_positives

        self.precisions.append(new_precision)
        self.recalls.append(new_recall)
        return 1.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_iou = self._calculate_intersect_over_union(y_true, y_pred)

        true_boxes = K.reshape(
            x=y_true[..., :5],
            shape=(-1, self.grid_rows * self.grid_cols, 5)
        )

        reshaped_batch_iou = reshape(batch_iou, shape=[-1])
        reshaped_true_conf = reshape(true_boxes[..., 0], shape=[-1])
        max_positives = tf.reduce_sum(reshaped_true_conf)
        tf.print(max_positives, output_stream=sys.stdout)

        map_fn(lambda x: self._classify_prediction(x[0], x[1], max_positives),
               [reshaped_batch_iou, reshaped_true_conf],
               dtype=tf.float32)
        #
        # new_precision = self.true_positives / (self.true_positives +
        #                                        self.false_positives)
        # new_recall = self.true_positives / (self.true_positives +
        #                                     self.false_negatives)
        # aux = new_precision if len(reshaped_true_conf) == self.grid_cols * self.grid_rows else self.last_precision
        # # tensorflow.print(self.last_precision, output_stream=sys.stdout)
        # # tensorflow.print(self.last_recall, output_stream=sys.stdout)
        # if new_precision < self.last_precision or len(reshaped_true_conf) == self.grid_rows * self.grid_cols:
        #     self.auc.assign_add(
        #         # self.last_precision * (new_recall - self.last_recall)
        #         aux * (new_recall - self.last_recall)
        #     )
        #     self.last_recall.assign(new_recall)
        # self.last_precision.assign(new_precision)


        tf.print(self.true_positives, output_stream=sys.stdout)
        tf.print(self.false_positives, output_stream=sys.stdout)
        tf.print(self.false_negatives, output_stream=sys.stdout)
        # tensorflow.print(self.last_precision, output_stream=sys.stdout)
        # tensorflow.print(self.last_recall, output_stream=sys.stdout)
        # tensorflow.print(self.auc, output_stream=sys.stdout)
        # if new_precision < self.last_precision or\
        #         len(reshaped_true_conf) == 1:
        #     # append old precision and recall to lists
        #     self.precisions.append(self.last_precision.value())
        #     self.recalls.append(self.last_recall.value())
        #
        # # assign new precision and recall
        # self.last_precision.assign(new_precision)
        # self.last_recall.assign(new_recall)

    @tf.function
    def aux(self):
        # indices = tf.where(self.recalls > recall_level)
        prec_at_rec = []
        for recall_value in tf.linspace(0.0, 1.0, 11):
            max_val = 0.0
            for j in range(0, len(self.precisions)):
                if self.recalls[j] >= recall_value and self.precisions[j] > max_val:
                    max_val = self.precisions[j]
            prec_at_rec.append(max_val)
        # if tf.shape(indices)[0] == 0:
        #     return tf.constant(0.0)
        return prec_at_rec

    def result(self):
        # accumulator = 0
        # # with Session() as sess:
        # #     aux = sess.run(self.last_used_recall.value())
        # for i in range(0, len(self.precisions)):
        #     if i == 0:
        #         recall_diff = self.recalls[i] - self.last_used_recall
        #     else:
        #         recall_diff = self.recalls[i] - self.recalls[i - 1]
        #
        #     accumulator += (self.precisions[i] * recall_diff)
        #
        # self.last_used_recall.assign(self.recalls[-1])
        # self.precisions = []
        # self.recalls = []
        #
        # self.auc.assign_add(accumulator)
        #
        # return self.auc.value()
        # prec_at_rec = tf.map_fn(lambda x: self.aux(x), tf.constant([0.0]))

        return tf.reduce_mean(self.aux())


class F1Score(Metric):
    def __init__(self, iou_threshold, grid_size, **kwargs):
        super(F1Score, self).__init__(name="F1Score", **kwargs)
        self.iou_threshold = iou_threshold
        self.grid_rows = grid_size[0]
        self.grid_cols = grid_size[1]

        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

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
                self.false_positives.assign_add(1)
        else:
            if true_confidence == 1.0:
                self.false_negatives.assign_add(1)

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
               dtype=tf.float32)

    def result(self):
        recall = self.true_positives / (self.true_positives +
                                        self.false_negatives)
        precision = self.true_positives / (self.true_positives +
                                           self.false_positives)
        return 2 * (precision * recall) / (precision + recall)
