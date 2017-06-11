import numpy as np
from keras.models import Model
from keras.layers import Input, Reshape, Flatten
from keras.layers import Dense, BatchNormalization, Activation, LeakyReLU, Dropout
from keras.layers import Conv2D, Conv2DTranspose


class GAN(object):
    """TODO: Write me.
    Args:

    Methods:
    """

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

    def _build_g_model(self):

        x = Input(shape=(self.g_input_dim, ))

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

        self.g_model = Model(x, y, name='generator')

    def _build_a_model(self):

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

        self.a_model = Model(x, y, name='adversarial_discriminator')
        self.a_model.compile(loss=self.a_loss, optimizer=self.a_optimizer, metrics=['accuracy'])

    def _build_stacked_model(self):
        x = Input(shape=(self.g_input_dim, ))
        y = self.a_model(self.g_model(x))

        self.stacked_model = Model(x, y, name='stacked_model')
        self.stacked_model.compile(loss=self.ga_loss, optimizer=self.ga_optimizer, metrics=['accuracy'])

    def build(self):
        """Builds GAN model.
        """
        self._build_g_model()
        self._build_a_model()
        self._build_stacked_model()

    def summary(self):
        """Prints summary of models to stdout.
        """
        print('Generator model:')
        self.g_model.summary()

        print('Adversarial discriminator model:')
        self.a_model.summary()

        print('Stacked model:')
        self.stacked_model.summary()

    def train_on_batch(self, x_real):
        """Trains GAN on a batch of samples.
        Args:
            x_real: Batch of samples.

        Returns:
            A tuple of:
                - Metrics of adversarial discriminator training update.
                - Metrics of stacked model training update.

        """
        batch_size = x_real[0]

        # Produce fake images with generator and concatenate with a batch of real images
        noise = np.random.rand(batch_size, self.g_input_dim)
        x_fake = self.g_model.predict(noise)

        x = np.concatenate((x_real, x_fake), axis=0)
        y = np.ones((2*batch_size, 1))
        y[batch_size:] = 0

        # Perform a batch update on the discriminator
        a_model_metrics = self.a_model.train_on_batch(x, y)

        # Freeze discriminator weights
        self.set_trainability(self.a_model, False)

        # Perform a batch update on the stacked model feeding noise and forced "real" labels.
        x = np.random.rand(batch_size, self.g_input_dim)
        y = np.ones((batch_size, 1))
        stacked_model_metrics = self.stacked_model.train_on_batch(x, y)

        # Un-freeze discriminator weights
        self.set_trainability(self.a_model, True)

        return a_model_metrics, stacked_model_metrics

    def generate(self, samples=1):
        """Generate samples using generator model.
        Args:
            samples: Number of samples to generate.

        Returns:
            Generated samples.

        """
        noise = np.random.rand(samples, self.g_input_dim)
        return self.g_model.predict(noise)

    def discriminate(self, x):
        """Predicts Real/Fake for a batch of samples.
        Args:
            x: Batch of samples.

        Returns:
            Real/Fake probabilities.
        """
        return self.a_model.predict(x)

    @staticmethod
    def set_trainability(model, flag):
        model.trainable = flag
        for layer in model.layers:
            layer.trainable = flag

