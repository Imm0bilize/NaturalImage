from typing import Union

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor, EagerTensor]
Dataset = tf.data.Dataset
