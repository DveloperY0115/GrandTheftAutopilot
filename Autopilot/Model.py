"""
NN model
"""
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Add, ReLU, Lamda
from tensorflow.keras.models import Model

from training.utils import INPUT_SHAPE

# relu_bn for resnet
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

# residual_block for resnet
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3):
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def build_model(args):
    # image model
    img_input = Input(shape=INPUT_SHAPE)
    img_model = (Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))(img_input)
    img_model = (Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Dropout(args.keep_prob))(img_model)
    img_model = (Flatten())(img_model)
    img_model = (Dense(100, activation='elu'))(img_model)

    # image model implemented with resnet
    img_input = Input(shape=INPUT_SHAPE)
    num_filters = 16

    resnet_model = BatchNormalization()(img_input)
    resnet_model = Conv2D(kernel_size=3, strides=1,
                        filters=num_filters,
                        padding="same")(resnet_model)
    resnet_model = relu_bn(resnet_model)
    resnet_model = residual_block(resnet_model, downsample=True, filters=num_filters)
    num_filters *= 2
    resnet_model = residual_block(resnet_model, downsample=True, filters=num_filters)
    num_filters *= 2
    resnet_model = residual_block(resnet_model, downsample=True, filters=num_filters)
    # resnet_model = residual_block(resnet_model, downsample=True, filters=num_filters)
    resnet_model = AveragePooling2D(4)(resnet_model)
    resnet_model = Flatten()(resnet_model)
    resnet_model = Dense(100,activation='relu')(resnet_model)

    # radar model
    radar_input = Input(shape=RADAR_SHAPE)
    radar_model = (Conv2D(32, (5, 5), activation='elu'))(radar_input)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    radar_model = (Conv2D(64, (5, 5), activation='elu'))(radar_model)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    radar_model = (Dropout(args.keep_prob / 2))(radar_model)
    radar_model = (Flatten())(radar_model)
    radar_model = (Dense(10, activation='elu'))(radar_model)

    # speed
    speed_input = Input(shape=(1,))

    # combined model
    out = Concatenate()([img_model, radar_model])
    out = (Dense(50, activation='elu'))(out)
    out = Concatenate()([out, speed_input])
    out = (Dense(10, activation='elu'))(out)
    out = (Dense(1))(out)

    final_model = Model(inputs=[img_input, radar_input, speed_input], outputs=out)

    final_model = img_model
    final_model.summary()

    return final_model
