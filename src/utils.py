import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from typing import Union


Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor, EagerTensor]
Dataset = tf.data.Dataset
