from typing import Tuple, List

import tensorflow as tf

from utils import Dataset, Tensor
from config import IMG_SIZE


class DatasetGenerator:
    def __init__(self, paths_to_data: List[str], batch_size: int, is_train: bool, seed: int):
        self.paths_to_data: List[str] = paths_to_data
        self.ds_len: int = len(self.paths_to_data)

        if not self.ds_len:
            assert ValueError('Paths contains zero file')

        self.batch_size: int = batch_size
        self.is_train: bool = is_train
        self.seed = seed
        self._class_name: Tensor = tf.convert_to_tensor([b'airplane', b'car', b'cat', b'dog',
                                                         b'flower', b'fruit', b'motorbike', b'person'])

        self._n_classes: int = len(self._class_name)
        self._autotune = tf.data.experimental.AUTOTUNE

    def __len__(self) -> int:
        return self.ds_len

    @tf.function
    def _augmentation(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        x: Tensor = tf.image.random_flip_left_right(x, seed=self.seed)
        x: Tensor = tf.image.random_flip_up_down(x, seed=self.seed)
        x: Tensor = tf.image.random_brightness(x, max_delta=0.1, seed=self.seed)
        return x, y

    @tf.function
    def _normalize(self, x: Tensor) -> Tensor:
        x: Tensor = tf.cast(x, dtype=tf.float32)
        x: Tensor = x / 255.0
        return x

    def _to_one_hot(self, y: Tensor) -> Tensor:
        y: Tensor = tf.one_hot(y, depth=self._n_classes)
        y: Tensor = tf.squeeze(y)
        return y

    def _load_file(self, path: str) -> Tuple[Tensor, Tensor]:
        file: str = tf.io.read_file(path)
        file: Tensor = tf.io.decode_jpeg(file, channels=3)
        file: Tensor = tf.image.resize(file, [IMG_SIZE, IMG_SIZE])

        class_number: Tensor = tf.where(
            self._class_name == tf.strings.split(path, sep='/')[2]   # 2 - folder with class name in path
        )

        return self._normalize(file), self._to_one_hot(class_number)

    def get_dataset(self) -> Dataset:
        dataset: Dataset = tf.data.Dataset.from_tensor_slices(self.paths_to_data)
        dataset: Dataset = dataset.map(lambda x: self._load_file(x),
                                       num_parallel_calls=self._autotune)

        if self.is_train:
            dataset: Dataset = dataset.shuffle(buffer_size=self.ds_len, seed=self.seed)
            dataset: Dataset = dataset.map(lambda x, y: self._augmentation(x, y),
                                           num_parallel_calls=self._autotune)

        dataset: Dataset = dataset.batch(self.batch_size)
        dataset: Dataset = dataset.repeat()
        return dataset
