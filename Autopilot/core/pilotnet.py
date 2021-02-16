import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class pilotNet:

    def __init__(self, drop_out_rate):
        """
        Create and initialize pilotNet for vehicle control
        """
        self.drop_out_rate = drop_out_rate
        self.model = models.Sequential()
        self.model.add(Rescaling(scale=1./255))

        # three Conv2D layers with 5 x 5 kernels, and 2 x 2 strides
        self.model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),
                              padding='valid', activation='relu', input_shape=(1, 200, 66, 3)))
        self.model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2),
                              padding='valid', activation='relu'))
        self.model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),
                              padding='valid', activation='relu'))

        # two Conv2D layers with 3 x 3 kernels, and no strides
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3),
                              padding='valid', activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3),
                              padding='valid', activation='relu'))

        # and data flows to three fully-connected layers
        self.model.add(Flatten())   # (None, 1152)
        self.model.add(Dense(units=1164))
        self.model.add(Dropout(rate=self.drop_out_rate))
        self.model.add(Dense(units=100))
        self.model.add(Dropout(rate=self.drop_out_rate))
        self.model.add(Dense(units=50))
        self.model.add(Dropout(rate=self.drop_out_rate))
        self.model.add(Dense(units=10))
        self.model.add(Dropout(rate=self.drop_out_rate))
        self.model.add(Dense(units=1))

        # build the pilotNet model
        self.model.build(input_shape=(1, 200, 66, 3))

        self.model.summary()


    def __call__(self, *args, **kwargs):
        """
        An alternative to call function 'predict' via Python special method
        :param args:
        :param kwargs:
        :return:
        """
        return self.predict(args, kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict the steering angle based on the input image
        :param args:
        :param kwargs:
        :return:
        """

        return None
