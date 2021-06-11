from typing import List

import tensorflow as tf

from models import Model
from dataset_generator import DatasetGenerator
from utils import Tensor, Dataset
from config import PATH_TO_DATA, VAL_SPLIT, LR_RATE, BATCH_SIZE, \
                   N_CLASSES, N_EPOCHS, SEED, PATH_TO_TRAINED_WEIGHTS


model: tf.keras.Model = Model(n_classes=N_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR_RATE)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()


def train_step(x: Tensor, y: Tensor) -> Tensor:
    with tf.GradientTape() as tape:
        y_pred: Tensor = model(x, training=True)
        loss: Tensor = tf.keras.losses.categorical_crossentropy(y, y_pred)
    grads: Tensor = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, y_pred)
    return loss


def val_step(x: Tensor, y: Tensor) -> Tensor:
    y_pred: Tensor = model(x, training=False)
    loss: Tensor = tf.keras.losses.categorical_crossentropy(y, y_pred)
    val_acc_metric.update_state(y, y_pred)
    return loss


def start_train(train_ds: Dataset, train_steps: int, val_ds: Dataset, val_steps: int) -> None:
    for epoch in range(N_EPOCHS):
        print(f"Start of training epoch: {epoch}")
        for batch_idx, (x, y) in enumerate(train_ds):
            train_loss: Tensor = train_step(x, y)
            print(f"Train loss: {tf.math.reduce_mean(train_loss)}\t"
                  f"Train accuracy: {train_acc_metric.result().numpy()}")

            if not batch_idx % train_steps and not batch_idx:
                for val_batch_idx, (val_x, val_y) in enumerate(val_ds):
                    val_loss: Tensor = train_step(val_x, val_y)
                    print(f"Val loss: {tf.math.reduce_mean(val_loss)}\t"
                          f"Val accuracy: {val_acc_metric.result().numpy()}")
                    if not val_batch_idx % val_steps and not val_batch_idx:
                        break


def main() -> None:
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    all_paths: List[str] = tf.io.gfile.glob(f'{PATH_TO_DATA}/*/*.jpg')
    all_paths: List[str] = tf.random.shuffle(all_paths, seed=SEED)

    val_paths: List[str] = all_paths[:int(len(all_paths)*VAL_SPLIT)]
    val_ds: Dataset = DatasetGenerator(val_paths, BATCH_SIZE, True, SEED)
    val_steps: int = len(val_ds) // BATCH_SIZE

    train_paths: List[str] = all_paths[int(len(all_paths)*VAL_SPLIT):]
    train_ds: Dataset = DatasetGenerator(train_paths, BATCH_SIZE, True, SEED)
    train_steps: int = len(train_ds) // BATCH_SIZE

    start_train(train_ds.get_dataset(), train_steps,val_ds.get_dataset(), val_steps)

    model.save(PATH_TO_TRAINED_WEIGHTS)
    print(f'Models weights saved in {PATH_TO_TRAINED_WEIGHTS}')


if __name__ == '__main__':
    main()
