import tensorflow as tf


Dataset = [
    tf.python.data.ops.dataset_ops.TensorSliceDataset,
    tf.python.data.ops.dataset_ops.ParallelMapDataset,
    tf.python.data.ops.dataset_ops.BatchDataset,
    tf.python.data.ops.dataset_ops.RepeatDataset
]