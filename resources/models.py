from keras.models import Model
from keras.layers import Input, Reshape, Flatten
from keras.layers import Dense, BatchNormalization, Activation, LeakyReLU, Dropout
from keras.layers import Conv2D, Conv2DTranspose


def set_trainability(model, flag):
    model.trainable = flag
    for layer in model.layers:
        layer.trainable = flag


class GAN:

    def __init__(self,
                 g_input_dim=100, g_filters=256, g_kernel_size=5,
                 a_filters=256, a_kernel_size=5, a_leakyrelu_alpha=0.2, a_dropout_rate=0.5,
                 a_loss='binary_crossentropy', a_optimizer='adam',
                 ga_loss='binary_crossentropy', ga_optimizer='adam'):

        # Generative model parameters
        self.g_input_dim = g_input_dim
        self.g_filters = g_filters
        self.g_kernel_size = g_kernel_size

        # Adversarial discriminator model parameters
        self.a_filters = a_filters
        self.a_kernel_size = a_kernel_size
        self.a_leakyrelu_alpha = a_leakyrelu_alpha
        self.a_dropout_rate = a_dropout_rate
        self.a_loss = a_loss
        self.a_optimizer = a_optimizer

        # Generative Adversarial model (stacked model) parameters
        self.ga_loss = ga_loss
        self.ga_optimizer = ga_optimizer

    def _build_g_net(self):

        x = Input(shape=(self.g_input_dim,))

        y = Dense(7 * 7 * self.g_filters)(x)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
        y = Reshape((7, 7, self.g_filters))(y)

        y = Conv2DTranspose(self.g_filters // 2, self.g_kernel_size, strides=(2, 2), padding='same')(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)

        y = Conv2DTranspose(self.g_filters // 4, self.g_kernel_size, strides=(2, 2), padding='same')(y)
        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)

        y = Conv2D(1, (1, 1), padding='same')(y)
        y = Activation('sigmoid')(y)

        self.g_net = Model(x, y)
        self.g_net.summary()

    def _build_a_net(self):

        x = Input(shape=(28, 28, 1))

        y = Conv2D(self.a_filters // 8, self.a_kernel_size, strides=(2, 2), padding='same')(x)
        y = LeakyReLU(alpha=self.a_leakyrelu_alpha)(y)
        y = Dropout(self.a_dropout_rate)(y)

        y = Conv2D(self.a_filters // 4, self.a_kernel_size, strides=(2, 2), padding='same')(y)
        y = LeakyReLU(alpha=self.a_leakyrelu_alpha)(y)
        y = Dropout(self.a_dropout_rate)(y)

        y = Conv2D(self.a_filters // 2, self.a_kernel_size, strides=(2, 2), padding='same')(y)
        y = LeakyReLU(alpha=self.a_leakyrelu_alpha)(y)
        y = Dropout(self.a_dropout_rate)(y)

        y = Conv2D(self.a_filters, self.a_kernel_size, strides=(2, 2), padding='same')(y)
        y = LeakyReLU(alpha=self.a_leakyrelu_alpha)(y)
        y = Dropout(self.a_dropout_rate)(y)

        y = Flatten()(y)
        y = Dense(1)(y)
        y = Activation('sigmoid')(y)

        self.a_net = Model(x, y)
        self.a_net.summary()

    def _build_stacked_net(self):
        self.stacked_net = self.a_net(self.g_net)
        self.stacked_net.summary()
