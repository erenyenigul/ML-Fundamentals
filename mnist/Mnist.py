import numpy as np


class Mnist :
    def __init__(self,images_dir,labels_dir):
        self.img_dir =images_dir
        self.lad_dir = labels_dir

    def get_images(self, rng):
        f = open(self.img_dir, "rb")
        image_size = 28
        num_images = rng[1]-rng[0]
        f.read(16+rng[0]*image_size*image_size)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data.reshape(num_images, image_size ** 2)

    def get_labels(self, rng):
        f = open(self.lad_dir, "rb")
        f.read(8+rng[0])
        buf = f.read(rng[1]-rng[0])
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

