from tensorflow import keras
from tensorflow.keras import layers


class DoubleConv(keras.Model):
    def __init__(self, n_filter, **kwargs):
        super(DoubleConv, self).__init__(**kwargs)
        self.n_filter = n_filter
        self.conv_dict = dict(kernel_size=(3, 3), activation="relu",
                              padding="same", kernel_initializer="he_uniform")

    def call(self, inputs):
        x = layers.Conv2D(self.n_filter, **self.conv_dict)(inputs)
        x = layers.Conv2D(self.n_filter, **self.conv_dict)(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        return x


class TripleConv(keras.Model):
    def __init__(self, n_filter, **kwargs):
        super(TripleConv, self).__init__(**kwargs)
        self.n_filter = n_filter
        self.conv_dict = dict(kernel_size=(3, 3), activation="relu",
                              padding="same", kernel_initializer="he_uniform")

    def call(self, inputs):
        x = layers.Conv2D(self.n_filter, **self.conv_dict)(inputs)
        x = layers.Conv2D(self.n_filter, **self.conv_dict)(x)
        x = layers.Conv2D(self.n_filter, **self.conv_dict)(x)
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
        return x


class Model(keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.d_conv_1 = DoubleConv(n_filter=64)
        self.d_conv_2 = DoubleConv(n_filter=128)
        self.t_conv_1 = TripleConv(n_filter=256)
        self.t_conv_2 = TripleConv(n_filter=512)
        self.t_conv_3 = TripleConv(n_filter=512)

    def call(self, inputs):
        x = self.d_conv_1(inputs)
        x = self.d_conv_2(x)
        x = self.t_conv_1(x)
        x = self.t_conv_2(x)
        x = self.t_conv_3(x)

        x = layers.Flatten()(x)

        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dense(4096, activation='relu')(x)
        x = layers.Dense(self.n_classes, activation='softmax')(x)
        return x
