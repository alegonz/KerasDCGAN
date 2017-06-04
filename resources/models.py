from keras.models import Model
from keras.layers import Input, Reshape, Flatten
from keras.layers import Dense, BatchNormalization, Activation, LeakyReLU, Dropout
from keras.layers import Conv2D, Conv2DTranspose


class GAN:

    def __init__(self,
                 g_input_dim=100, g_filters=256, g_kernel_size=5,
                 d_filters=256, d_kernel_size=5, d_leakyrelu_alpha=0.2, d_dropout_rate=0.5,
                 d_loss='binary_crossentropy', d_optimizer='adam',
                 am_loss='binary_crossentropy', am_optimizer='adam'):

        # Generative model parameters
        self.g_input_dim = g_input_dim
        self.g_filters = g_filters
        self.g_kernel_size = g_kernel_size
        self.g_loss = g_loss
        self.g_optimizer = g_optimizer

        # Discriminator model parameters
        self.d_filters = d_filters
        self.d_kernel_size = d_kernel_size
        self.d_leakyrelu_alpha = d_leakyrelu_alpha
        self.d_dropout_rate = d_dropout_rate
        self.d_loss = d_loss
        self.d_optimizer = d_optimizer

    def _build_g(self):

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

        self.G = Model(x, y)
        print(self.G.summary())

    def _build_d(self):

        x = Input(shape=(28, 28, 1))

        y = Conv2D(self.d_filters // 8, self.d_kernel_size, strides=(2, 2), padding='same')(x)
        y = LeakyReLU(alpha=self.d_leakyrelu_alpha)(y)
        y = Dropout(self.d_dropout_rate)(y)

        y = Conv2D(self.d_filters // 4, self.d_kernel_size, strides=(2, 2), padding='same')(y)
        y = LeakyReLU(alpha=self.d_leakyrelu_alpha)(y)
        y = Dropout(self.d_dropout_rate)(y)

        y = Conv2D(self.d_filters // 2, self.d_kernel_size, strides=(2, 2), padding='same')(y)
        y = LeakyReLU(alpha=self.d_leakyrelu_alpha)(y)
        y = Dropout(self.d_dropout_rate)(y)

        y = Conv2D(self.d_filters, self.d_kernel_size, strides=(2, 2), padding='same')(y)
        y = LeakyReLU(alpha=self.d_leakyrelu_alpha)(y)
        y = Dropout(self.d_dropout_rate)(y)

        y = Flatten()(y)
        y = Dense(1)(y)
        y = Activation('sigmoid')(y)

        self.D = Model(x, y)
        print(self.D.summary())
