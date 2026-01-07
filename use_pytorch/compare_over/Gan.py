

from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.layers import Input,Dense
from keras.models import Model, Sequential
import pandas as pd
import numpy as np
number = 426

feature = 4

path = 'balance'

n1=64

n2=32

ld=70

class GAN():
    def __init__(self):
        # self.rows = 1
        # self.cols =feature
        # self.channel =1 信道数量定义
        self.data_shape = feature
        self.latent_dim = ld

        optimizer = Adam()

        #构造和编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss ='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #构造生成器
        self.generator = self.build_generator()

        #生成器将噪声作为输入，然后生成数据
        z = Input(shape = (self.latent_dim,))
        jia = self.generator(z)

        #对于整体的模型，我们只训练生成器
        self.discriminator.trainable =False

        #判别器将生成的数据作为输入然后判别真假
        validity = self.discriminator(jia)

        #整体模型的构造，训练生成器来混淆判别器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(n1, input_dim =self.latent_dim, activation='relu'))
        # model.add(Dropout =0.5)
        model.add(Dense(n1, activation='relu'))
        model.add(Dense(n1, activation='relu'))
        model.add(Dense(self.data_shape, activation='relu'))

        noise =Input(shape=(self.latent_dim,))
        data = model(noise)

        return Model(noise, data)

    def build_discriminator(self):
        model =Sequential()
        model.add(Dense(n2, input_dim=self.data_shape, activation='relu'))
        model.add(Dense(n2, activation='relu'))
        model.add(Dense(n2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        data =Input(shape=(self.data_shape, ))
        validity = model(data)

        return Model(data, validity)

    def train(self, epochs, batch_size =32, sample_interval=100):



        dataframe =pd.read_csv("E:/imbalanced data/Dataset/%s/rtrain.csv"%(str(path)), header=None)

        dataframe =shuffle(dataframe)
        dataset = dataframe.values

        X_train = dataset[:, 0:feature]

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of datas
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            datas = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new datas
            gen_datas = self.generator.predict(noise)

            # Train the discriminator

            d_loss_real = self.discriminator.train_on_batch(datas, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_datas, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # print(d_loss.shape)
            # print(d_loss)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated data samples
            if epoch % sample_interval == 0:
                self.generator.save_weights('generativesave/gangenerator', overwrite=True)
                # self.sample_images(epoch)

    def generate_images(self):

        self.generator.load_weights('generativesave/gangenerator')  # cgantest
        # gen_data = np.zeros(1,12)
        gen_data1 = []
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
        data.to_csv('E:/traffic accident/Myself/generativedata/gan.csv', index=False, header=False)

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=5000, batch_size=32, sample_interval=100)
    gan.generate_images()
