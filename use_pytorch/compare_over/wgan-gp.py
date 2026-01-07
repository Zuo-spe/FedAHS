# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from functools import partial
from  sklearn.utils import  shuffle

import pandas as pd

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

number = 426

feature = 4

path = 'balance'

n1=64

n2=32

ld=70

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        # self.data_rows = 1
        # self.data_cols = 12
        # self.channels = 1
        self.data_shape = feature
        # self.latent_dim = 9， 以前和生成样本的维度一致，后来发现可以不一致，gan文献里面是100维，cgan10~130不定
        self.latent_dim = ld

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        # optimizer = RMSprop(lr=0.00005)#Adam
        optimizer = Adam(lr=0.0001, beta_1=0, beta_2=0.9)
        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # data input (real sample)
        real_data = Input(shape=(self.data_shape,))

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))

        # Generate image based of noise (fake sample)
        fake_data = self.generator(z_disc)
        print(real_data.shape)
        print(fake_data.shape)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_data)
        valid = self.critic(real_data)

        # Construct weighted average between real and fake images
        interpolated_data = RandomWeightedAverage()([real_data, fake_data])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_data)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_data)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_data, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        data = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(data)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):

        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(n1, input_dim=self.latent_dim, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))

        # model.add(Dense(64, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.5))

        model.add(Dense(n1, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(n1, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.data_shape), activation='relu'))
        # model.summary()

        noise = Input(shape=(self.latent_dim,))

        data = model(noise)

        return Model(noise, data)

    def build_critic(self):

        model = Sequential()
        model.add(Dense(n2, input_dim=self.data_shape, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.5))
        model.add(Dense(n2, activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dropout(0.5))
        model.add(Dense(n2, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(1))

        # model.summary()

        data = Input(shape=(self.data_shape,))
        validity = model(data)

        return Model(data, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        dataframe = pd.read_csv("E:/imbalanced data/Dataset/%s/train.csv"%(str(path)), header=None)

        dataframe = shuffle(dataframe)
        dataset = dataframe.values

        X_train = dataset[:, 0:feature].astype(float)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                datas = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([datas, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # self.sample_images(epoch)
               self.generator.save_weights('generativesave/wgan-gpgenerator', overwrite=True)
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def generate_images(self):

        self.generator.load_weights('generativesave/wgan-gpgenerator')#wgan-gpgenerator
        # gen_data = np.zeros(1,12)
        gen_data1 =[]
        # print(gen_data.shape)
        for i in range(number):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_data1.append(self.generator.predict(noise))
            # gen_imgs = np.squeeze(gen_imgs)
            # gen_imgs = gen_imgs.reshape(1, -1)
            # print(gen_imgs)
            # print(gen_imgs1.shape)
            # print(gen_data1.shape)
            # gen_data = np.concatenate((gen_data1, gen_data))
        gen_data1 = np.array(gen_data1)
        gen_data1 = np.squeeze(gen_data1)
        data = pd.DataFrame(gen_data1)
        data.to_csv('E:/traffic accident/Myself/generativedata/wgan-gp.csv', index=False, header=False)


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=5000, batch_size=32, sample_interval=100)
    wgan.generate_images()