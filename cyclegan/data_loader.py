from imageio import imread
from glob import glob
import numpy as np
from PIL import Image


class DataLoader:
    def __init__(self, dataset_name, img_res, load_path):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.load_path = load_path

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob(self.load_path+'/datasets/%s/%s/*' % (self.dataset_name, data_type))
        batch_images = np.random.choice(path, size=batch_size)
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = Image.fromarray(img).resize(self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = Image.fromarray(img).resize(self.img_res)
            img = np.array(img)
            imgs.append(img)
        imgs = np.array(imgs) / 255

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(self.load_path+'/datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob(self.load_path+'/datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = Image.fromarray(img_A).resize(self.img_res)
                img_B = Image.fromarray(img_B).resize(self.img_res)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                img_A = np.array(img_A)
                img_B = np.array(img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 255
            imgs_B = np.array(imgs_B) / 255

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = Image.fromarray(img).resize(self.img_res)
        img = np.array(img)
        img = img / 255
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return imread(path, as_gray=False, pilmode="RGB").astype(np.uint8)
