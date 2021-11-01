"""Image generator implementation. next() function returns the next generated object. It consists of a batch of
images and its corresponding labels.
"""
import os
import json
import random
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_dir = file_path
        self.cur_batch = 0
        self.cur_epoch = 0
        self.class_dict = {
                            0: 'airplane',
                            1: 'automobile',
                            2: 'bird',
                            3: 'cat',
                            4: 'deer',
                            5: 'dog',
                            6: 'frog',
                            7: 'horse',
                            8: 'ship',
                            9: 'truck'
                        }
        self.class_labels = dict()
        with open(label_path) as f:
            self.class_labels = json.load(f)
        self.list_imgs = os.listdir(file_path)
        self.total_imgs = len(self.list_imgs)
        self._batch_generator = self._get_next_batch()
        self.shuffle = shuffle
        self.mirroring = mirroring
        self.rotation = rotation
        self.rotation_angles = [0,1,2,3,4] #multiplied by 90deg
        self.mirror_axis = [0,1]
        if self.shuffle:
            random.shuffle(self.list_imgs)

    def next(self):
        """This function creates a batch of images and corresponding labels and returns them.
        """
        batch = next(self._batch_generator)
        images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), dtype=np.float64)
        labels = []
        for item in range(self.batch_size):
            img= np.load(os.path.join(self.data_dir, batch[item]))
            img = resize(img, (self.image_size[1], self.image_size[0]))
            images[item] = self.augment(img)
            labels.append(self.class_labels[batch[item].split(".")[0]])
        return images, labels

    def _get_next_batch(self):
        counter = 0
        cur_index = 0
        batch = []
        while True:
            if counter == self.batch_size:
                counter = 0
                yield batch
                batch = []
                self.cur_batch += 1
            if cur_index == self.total_imgs:
                cur_index = 0
                self.cur_epoch += 1
                if self.shuffle:
                    random.shuffle(self.list_imgs)
            batch.append(self.list_imgs[cur_index])
            counter += 1
            cur_index += 1


    def augment(self,img):
        """this function takes a single image as an input and performs a random transformation
        (mirroring and/or rotation) on it and outputs the transformed image
        """
        if self.rotation:
            img = np.rot90(img, random.choice(self.rotation_angles), (0,1))
        if self.mirroring:
            img = np.flip(img, random.choice(self.mirror_axis))
        return img

    def current_epoch(self):
        """return the current epoch number
        """
        return self.cur_epoch

    def class_name(self, x):
        """This function returns the class name for a specific label
        """
        return self.class_dict[x]

    def show(self, headless=False):
        images, labels = self.next()
        cols = 4
        rows = int(np.ceil(self.batch_size/4))
        fig = plt.figure(figsize=(8, 8))
        for item in range(len(labels)):
            fig.add_subplot(rows, cols, item+1)
            plt.axis('off')
            plt.title(self.class_name(labels[item]))
            plt.imshow(images[item])
        if headless:
            plt.savefig("generator.png")
        else:
            plt.show()
