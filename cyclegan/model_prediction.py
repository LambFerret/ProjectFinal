# -*- coding: utf-8 -*-

import keras
from keras.optimizers import Adam
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader3 import DataLoader
from glob import glob

# test A에 넣으셔야 합니다
class Predictions:
    def __init__(self, dataset, loadnum, isColab = True):

        if not isColab:
            self.load_path = "."
        else:
            self.load_path = "./drive/MyDrive"

        self.img_rows = 256
        self.img_cols = 256

        self.load_number = loadnum
        self.dataset_name = dataset
        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle

        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        self.combined = keras.models.load_model(self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/combined")
        print("combined loaded in :" + self.load_path)
        self.d_A = keras.models.load_model(self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/dA")
        print("d_A loaded")
        self.d_B = keras.models.load_model(self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/dB")
        print("d_B loaded")
        self.g_AB = keras.models.load_model(self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/gAB")
        print("g_AB loaded")
        self.g_BA = keras.models.load_model(self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/gBA")
        print("g_BA loaded")
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=Adam(0.0002, 0.5))


    def sample_images(self):
        # os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

        inputPic = glob(self.load_path + "/datasets/" + self.dataset_name+"/inputPic/*")

        imgs_A = self.data_loader.load_img(inputPic[0])
        fake_B = self.g_AB.predict(imgs_A)
        reconstr_A = self.g_BA.predict(fake_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A])

        titles = ['Original', 'Translated', 'Reconstructed']
        for a in range(3):
            plt.imshow(gen_imgs[a])
            plt.savefig(f"{self.load_path}/images/result_{titles[a]}")
            plt.close()


if __name__ == '__main__':
    predict = Predictions(dataset='spring2fall', loadnum=1, isColab=True)
    predict.sample_images()
