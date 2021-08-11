# -*- coding: utf-8 -*-

from keras import models
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os, sys
from data_loader import DataLoader


class Predictions:
    def __init__(self, dataset, loadnum, isColab=True):

        if not isColab:
            self.load_path = "."
        else:
            self.load_path = "./drive/MyDrive"

        self.img_rows = 256
        self.img_cols = 256

        self.load_number = loadnum
        self.dataset_name = dataset
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

        if not glob(self.load_path + "/inputPic/*"):
            print("put A picture in inputPic")
            os.makedirs(self.load_path + '/inputPic', exist_ok=True)
            sys.exit()

        self.lambda_cycle = 10.0  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle

        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        self.g_AB = models.load_model(
            self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/gAB")
        print("g_AB loaded")
        self.g_BA = models.load_model(
            self.load_path + "/saved_model/" + self.dataset_name + str(self.load_number) + "/gBA")
        print("g_BA loaded")

    def sample_images(self):
        # os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

        inputPic = glob(self.load_path + "/inputPic/*")
        imgs_A = self.data_loader.load_img(inputPic[0])
        fake_B = self.g_AB.predict(imgs_A)
        reconstr_A = self.g_BA.predict(fake_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A])

        titles = ['Original', 'Translated', 'Reconstructed']

        for a in range(3):
            plt.imshow(gen_imgs[a])
            plt.savefig(f"{self.load_path}/images/result_{titles[a]}")
            print(f"{titles[a]} created!")
            plt.close()


if __name__ == '__main__':
    predict = Predictions(dataset='spring2fall', loadnum=0, isColab=False)
    predict.sample_images()
