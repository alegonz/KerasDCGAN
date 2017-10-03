import numpy as np
from keras.models import Model
from keras.layers import Input, Reshape, Flatten
from keras.layers import Dense, BatchNormalization, Activation, LeakyReLU, Dropout
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam, SGD


class DCGAN(object):
    """Deep Convolutional Generative Adversarial Network.

    Args:
        # Generator model:
        g_input_dim: Number of inputs.
        g_filters: Number of filters of convolution layers.
        g_kernel_size: Kernel size of convolution layers.
        g_dropout_rate: Dropout rate.
        g_bn_momentum: Batch normalization momentum.

        # Adversarial Discriminator model:
        a_filters: Number of filters of convolution layers.
        a_kernel_size: Kernel size of convolution layers.
        a_leakyrelu_alpha: Alpha parameter of LeakyReLU activation.
        a_dropout_rate: Dropout rate.
        a_bn_momentum: Batch normalization momentum.
        a_loss: Loss function.
        a_optimizer: Optimizer.

        # Generative Adversarial (stacked) model:
        ga_loss: Loss function.
        ga_optimizer: Optimizer.

    """

    def __init__(self,
                 g_input_dim=100, g_filters=256, g_kernel_size=5,
                 g_dropout_rate=0.4, g_bn_momentum=0.9, g_distribution='gaussian',
                 a_filters=512, a_kernel_size=5, a_leakyrelu_alpha=0.2,
                 a_dropout_rate=0.4, a_bn_momentum=0.9,
                 a_loss='binary_crossentropy', a_optimizer_lr=1e-5, a_optimizer_momentum=0.9,
                 ga_loss='binary_crossentropy', ga_optimizer_lr=1e-5):

        # Generative model parameters
        self.g_input_dim = g_input_dim
        self.g_filters = g_filters
        self.g_kernel_size = g_kernel_size
        self.g_dropout_rate = g_dropout_rate
        self.g_bn_momentum = g_bn_momentum
        self.g_distribution = g_distribution

        # Adversarial discriminator model parameters
        self.a_filters = a_filters
        self.a_kernel_size = a_kernel_size
        self.a_leakyrelu_alpha = a_leakyrelu_alpha
        self.a_dropout_rate = a_dropout_rate
        self.a_bn_momentum = a_bn_momentum
        self.a_loss = a_loss
        self.a_optimizer = SGD(lr=a_optimizer_lr, momentum=a_optimizer_momentum)

        # Generative Adversarial model (stacked model) parameters
        self.ga_loss = ga_loss
        self.ga_optimizer = Adam(lr=ga_optimizer_lr)

    def _build_g_model(self):

        x = Input(shape=(self.g_input_dim, ))

        y = Dense(7 * 7 * self.g_filters)(x)
        y = BatchNormalization(momentum=self.g_bn_momentum)(y)
        y = Activation('relu')(y)
        y = Reshape((7, 7, self.g_filters))(y)
        y = Dropout(self.g_dropout_rate)(y)

        y = Conv2DTranspose(self.g_filters // 2, self.g_kernel_size, strides=(2, 2), padding='same')(y)
        y = BatchNormalization(momentum=self.g_bn_momentum)(y)
        y = Activation('relu')(y)

        y = Conv2DTranspose(self.g_filters // 4, self.g_kernel_size, strides=(2, 2), padding='same')(y)
        y = BatchNormalization(momentum=self.g_bn_momentum)(y)
        y = Activation('relu')(y)

        y = Conv2DTranspose(self.g_filters // 8, self.g_kernel_size, strides=(1, 1), padding='same')(y)
        y = BatchNormalization(momentum=self.g_bn_momentum)(y)
        y = Activation('relu')(y)

        y = Conv2DTranspose(1, self.g_kernel_size, strides=(1, 1), padding='same')(y)
        y = Activation('tanh')(y)

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

        y = Conv2D(self.a_filters, self.a_kernel_size, strides=(1, 1), padding='same')(y)
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

    def generate_noise(self, n_samples):
        if self.g_distribution == 'uniform':
            noise = np.random.uniform(low=-1.0, high=1.0, size=(n_samples, self.g_input_dim))
        elif self.g_distribution == 'gaussian':
            noise = np.random.randn(n_samples, self.g_input_dim)
        else:
            raise ValueError('Invalid noise distribution.')
        return noise

    def generate(self, samples=1):
        """Generate samples using generator model.
        Args:
            samples: An integer indicating the number of samples to generate,
                or an array of precomputed random values.

        Returns:
            Generated samples.

        """
        if isinstance(samples, int):
            samples = self.generate_noise(samples)
        elif not isinstance(samples, np.ndarray):
            raise ValueError('n_samples must be either an integer or a numpy.ndarray.')

        return self.g_model.predict(samples)

    def discriminate(self, x):
        """Predicts Real/Fake for a batch of samples.
        Args:
            x: Batch of samples.

        Returns:
            Real/Fake probabilities.
        """
        return self.a_model.predict(x)

    def pretrain(self, x_real, batch_size, epochs):
        # TODO: Revise this function
        """Pre-trains the adversarial discriminator model on given real data and (crude) generated data.
        Args:
            x_real: Training (real) data.
            batch_size: Training batch size.
            epochs: Number of epochs.

        Returns:
            Metrics of adversarial discriminator pre-training.

        """
        n_samples = x_real.shape[0]

        # Add fake images.
        x_fake = self.generate(n_samples)
        x = np.concatenate((x_real, x_fake), axis=0)
        y = np.ones((2 * n_samples, 1))
        y[n_samples:] = 0

        history = self.a_model.fit(x, y, batch_size=batch_size, epochs=epochs)

        return history

    def train_on_batch(self, x_real, freeze_discriminator=False):
        """Trains GAN on a batch of samples.
        Args:
            x_real: Batch of real samples.
            freeze_discriminator: Freeze discriminator during training of stacked model (True) or not (False).

        Returns:
            A tuple of:
                - Metrics of adversarial discriminator training update.
                - Metrics of stacked model training update.

        """
        batch_size = x_real.shape[0]

        # Produce fake images with generator and concatenate with a batch of real images
        x_fake = self.generate(batch_size)
        x = np.concatenate((x_real, x_fake), axis=0)
        y = np.ones((2*batch_size, 1))
        y[batch_size:] = 0

        # Perform a batch update on the discriminator
        a_model_metrics = self.a_model.train_on_batch(x, y)

        # Freeze discriminator weights
        if freeze_discriminator:
            self.set_trainability(self.a_model, False)

        # Perform a batch update on the stacked model feeding noise and forced "real" labels.
        x = self.generate_noise(batch_size)
        y = np.ones((batch_size, 1))
        stacked_model_metrics = self.stacked_model.train_on_batch(x, y)

        # Un-freeze discriminator weights
        if freeze_discriminator:
            self.set_trainability(self.a_model, True)

        return a_model_metrics, stacked_model_metrics

    @staticmethod
    def set_trainability(model, flag):
        model.trainable = flag
        for layer in model.layers:
            layer.trainable = flag
