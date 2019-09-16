import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data
import cPickle
import random as random
from sklearn.datasets import make_moons
from scipy.io import loadmat
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as util_shuffle
import skimage.io

prefix = './Datas/'
def get_img(img_path, crop_h, resize_h):
    img=scipy.misc.imread(img_path).astype(np.float)
    # crop resize
    crop_w = crop_h
    #resize_h = 64
    resize_w = resize_h
    h, w = img.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])

    return np.array(cropped_image)/255.0

class mnist():
    def __init__(self, flag='conv', is_tanh = True, is_color = False, dataset = 'mnist', test_batch =False):
        if dataset == 'mnist':
            datapath = prefix + 'mnist'
        elif dataset == 'fashion_mnist':
            datapath = prefix + 'fashion_mnist'
        self.X_dim = 784 # for mlp
        self.z_dim = 100
        self.zc_dim = 32
        self.y_dim = 10
        self.size = 28 # for conv
        self.is_color = is_color
        self.test_batch = test_batch
        if is_color:
            self.channel = 3
        else:
            self.channel = 1 # for conv
        self.data = input_data.read_data_sets(datapath, one_hot=False)
        self.flag = flag
        self.is_tanh = is_tanh

    def __call__(self,batch_size):
        if self.test_batch:
            batch_imgs, y = self.data.validation.next_batch(batch_size)
        else:
            batch_imgs,y = self.data.train.next_batch(batch_size)
        if self.flag == 'conv':
            batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, 1))
        if self.is_tanh:
            if self.is_color:
                random_color = np.random.normal(0, 0.5, (batch_size, 1, 1, self.channel))
                batch_imgs = batch_imgs * random_color + batch_imgs - 1
                random_noise = np.random.normal(0, 0.2, (batch_size, self.size, self.size, self.channel))
                batch_imgs += random_noise
                batch_imgs = np.clip(batch_imgs, -1, 1)
            else:
                batch_imgs = batch_imgs*2 - 1
        return batch_imgs, y

    def data2fig(self, samples, nr = 4, nc = 4):
        if self.is_tanh:
            samples = (samples + 1)/2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if self.is_color:
                plt.imshow(sample.reshape(self.size,self.size,self.channel), cmap='Greys_r')
            else:
                plt.imshow(sample.reshape(self.size, self.size), cmap='Greys_r')
        return fig

class Cifar10():
    def __init__(self, flag='conv', is_tanh = False, test_batch = False, all_images = True):
        datapath = prefix + 'cifar-10-batches-py'
        self.X_dim = 3072  # for mlp
        self.z_dim = 200
        self.y_dim = 10
        self.zc_dim = 100
        self.size = 32  # for conv
        self.channel = 3  # for conv
        self.flag = flag
        self.is_tanh = is_tanh

        datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

        if test_batch == True:
            datafiles = ['test_batch']

        if all_images == True:
            datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

        def unpickle(f):
            fo = open(f, 'rb')
            d = cPickle.load(fo)
            fo.close()
            return d

        self.data = []
        self.labels = []

        for f in datafiles:
            d = unpickle(datapath+'/'+f)
            data = d["data"]
            labels = np.array(d["labels"])
            nsamples = len(data)
            for idx in range(nsamples):
                self.data.append(data[idx].reshape(self.channel, self.size, self.size).transpose(1, 2, 0))
                self.labels.append(labels[idx])

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.uint8)
        self.data /= 255.0
        if self.is_tanh:
            self.data = self.data * 2 - 1

        self.num_examples = len(self.data)

        self.pointer = 0

        self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(self.num_examples)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])

        return batch

    def __call__(self, batch_size, random_flip = True):
        if self.pointer + batch_size > self.num_examples:
            rest_num_examples = self.num_examples - self.pointer
            images_rest_part = self.data[self.pointer:self.num_examples]
            labels_rest_part = self.labels[self.pointer:self.num_examples]
            self.shuffle_data()
            self.pointer = batch_size - rest_num_examples
            images_new_part = self.data[0:self.pointer]
            labels_new_part = self.labels[0:self.pointer]
            batch_data = np.concatenate((images_rest_part, images_new_part), axis=0)
            if random_flip:
                batch_data = self._random_flip_leftright(batch_data)
            return batch_data, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            start = self.pointer
            self.pointer += batch_size
            batch_data = self.data[start:self.pointer]
            if random_flip:
                batch_data = self._random_flip_leftright(batch_data)
            return batch_data, self.labels[start:self.pointer]

    def data2fig(self, samples, nr=4, nc=4):
        if self.is_tanh:
            samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size, self.size, self.channel), cmap='Greys_r')
        return fig

    def data2fig2jpg(self, samples, labels, y = 1):
        dst_dir = '/media/sist316/KINGSTON/generated_images/cifar-real'
        num = 0
        if self.is_tanh:
            samples = (samples + 1) / 2
        for i, sample in enumerate(samples):
            if labels[i] == y:
                img_path = os.path.join(dst_dir, str(y) + '_' + str(num) + '.jpg')
                skimage.io.imsave(img_path, sample)
                num += 1


class SVHN():
    def __init__(self, is_tanh = False, test_batch = False, all_images = True):
        datapath = prefix + 'SVHN'
        self.X_dim = 3072  # for mlp
        self.z_dim = 128
        self.zc_dim = 8
        self.y_dim = 10
        self.size = 32  # for conv
        self.channel = 3  # for conv
        self.is_tanh = is_tanh

        datafiles = ['extra_32x32.mat', 'train_32x32.mat']

        if test_batch == True:
            datafiles = ['test_32x32.mat']

        if all_images == True:
            datafiles = ['train_32x32.mat', 'extra_32x32.mat', 'test_32x32.mat']

        self.data = []
        self.labels = []

        for f in datafiles:
            d = loadmat(datapath+'/'+f)
            data = d['X']
            labels = d['y']
            self.data.append(data)
            self.labels.append(labels)

        self.data = np.concatenate(self.data, axis=3)
        self.data = np.transpose(self.data, (3, 0, 1, 2)).astype(np.float32)

        self.labels = np.concatenate(self.labels)
        self.labels = np.array([x[0] if x[0] != 10 else 0 for x in self.labels])

        self.data /= 255.0
        if self.is_tanh:
            self.data = self.data * 2 - 1

        self.num_examples = len(self.data)

        self.pointer = 0

        self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(self.num_examples)
        self.data = self.data[indices]
        self.labels = self.labels[indices]


    def __call__(self, batch_size, random_flip = True):
        if self.pointer + batch_size > self.num_examples:
            rest_num_examples = self.num_examples - self.pointer
            images_rest_part = self.data[self.pointer:self.num_examples]
            labels_rest_part = self.labels[self.pointer:self.num_examples]
            self.shuffle_data()
            self.pointer = batch_size - rest_num_examples
            images_new_part = self.data[0:self.pointer]
            labels_new_part = self.labels[0:self.pointer]
            batch_data = np.concatenate((images_rest_part, images_new_part), axis=0)
            return batch_data, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            start = self.pointer
            self.pointer += batch_size
            batch_data = self.data[start:self.pointer]
            return batch_data, self.labels[start:self.pointer]

    def data2fig(self, samples, nr=4, nc=4):
        if self.is_tanh:
            samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size, self.size, self.channel), cmap='Greys_r')
        return fig

class CUB_200_2011():
    def __init__(self, is_tanh = True):
        datapath = prefix + 'CUB_200_2011/CUB_200_2011'
        self.z_dim = 512
        self.y_dim = 200
        self.size= 128
        self.channel = 3
        self.is_tanh = is_tanh

        def read_images_labels(listfile):
            # Get all the images and labels in directory/label/*.jpg
            files_and_labels = []
            with open(listfile, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')[1]
                    label = int(line.split('/')[0].split('.')[0])
                    files_and_labels.append((os.path.join(datapath, 'images', line), label - 1))

            filenames, labels = zip(*files_and_labels)
            filenames = list(filenames)
            labels = list(labels)

            filenames = np.array(filenames, dtype=np.str)
            labels = np.array(labels, dtype=np.uint8)

            return filenames, labels


        def read_bboxes(listfile):
            bboxes = []
            with open(listfile, 'r') as f:
                for line in f:
                    _, x, y, w, h = line.strip().split(' ')
                    bboxes.append([int(float(x)), int(float(y)), int(float(w)), int(float(h))])
            bboxes = np.array(bboxes, dtype=np.int32)
            return bboxes

        self.data, self.labels = read_images_labels(os.path.join(datapath, 'images.txt'))
        self.bboxes = read_bboxes(os.path.join(datapath, 'bounding_boxes.txt'))

        self.num_examples = len(self.data)

        self.pointer = 0

        self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(self.num_examples)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
        self.bboxes = self.bboxes[indices]

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def get_img(self, img_path, bbox):
        x, y, w, h = bbox
        img = scipy.misc.imread(img_path, mode = 'RGB').astype(np.float)
        # crop by bounding box
        cropped_image = img[y:y + h, x:x + w, :]
        cropped_image = scipy.misc.imresize(cropped_image, [self.size, self.size])
        cropped_image = np.array(cropped_image) / 255.0
        if self.is_tanh:
            cropped_image = cropped_image * 2 - 1

        return cropped_image

    def __call__(self, batch_size, random_flip = True):
        if self.pointer + batch_size > self.num_examples:
            rest_num_examples = self.num_examples - self.pointer
            path_rest_part = self.data[self.pointer:self.num_examples]
            labels_rest_part = self.labels[self.pointer:self.num_examples]
            bboxes_rest_part = self.bboxes[self.pointer:self.num_examples]
            self.shuffle_data()
            self.pointer = batch_size - rest_num_examples
            path_new_part = self.data[0:self.pointer]
            labels_new_part = self.labels[0:self.pointer]
            bboxes_new_part = self.bboxes[0:self.pointer]
            batch_path = np.concatenate((path_rest_part, path_new_part), axis=0)
            batch_bboxes = np.concatenate((bboxes_rest_part, bboxes_new_part), axis=0)
            batch_data = [self.get_img(img_path, box) for img_path, box in zip(batch_path, batch_bboxes)]
            batch_data = np.array(batch_data)
            if random_flip:
                batch_data = self._random_flip_leftright(batch_data)
            return batch_data, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            start = self.pointer
            self.pointer += batch_size
            batch_path = self.data[start:self.pointer]
            batch_bboxes = self.bboxes[start:self.pointer]
            batch_data = [self.get_img(img_path, box) for img_path, box in zip(batch_path, batch_bboxes)]
            batch_data = np.array(batch_data)
            if random_flip:
                batch_data = self._random_flip_leftright(batch_data)
            return batch_data, self.labels[start:self.pointer]

    def data2fig(self, samples, nr=4, nc=4):
        if self.is_tanh:
            samples = (samples + 1) / 2
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size, self.size, self.channel), cmap='Greys_r')
        return fig

class facescrub():
    def __init__(self, is_tanh = True, size = 64):
        prefix = "./Datas/"
        datapath = prefix + 'facescrub_aligned'
        self.z_dim = 512
        self.y_dim = 530
        self.size= size
        self.channel = 3
        self.is_tanh = is_tanh

        def read_images_labels():
            # Get all the images and labels in directory/label/*.jpg
            files_and_labels = []
            for class_label in glob(datapath + '/*'):
                for imgfile in glob(class_label + '/*.png'):
                    files_and_labels.append((imgfile, class_label.split('/')[-1]))

            filenames, labels = zip(*files_and_labels)
            filenames = list(filenames)
            labels = list(labels)

            cls_dict ={}
            label_set = set(labels)
            for cls_id, cls_name in enumerate(label_set):
                cls_dict[cls_name] = cls_id

            labels = [cls_dict[label] for label in labels]

            filenames = np.array(filenames, dtype=np.str)
            labels = np.array(labels, dtype=np.uint16)

            return filenames, labels

        self.data, self.labels = read_images_labels()

        self.num_examples = len(self.data)

        self.pointer = 0

        # np.random.seed(9)
        self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(self.num_examples)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def get_img(self, img_path):
        img = scipy.misc.imread(img_path, mode = 'RGB').astype(np.float)
        # crop by bounding box
        img = scipy.misc.imresize(img, [self.size, self.size])
        img = np.array(img) / 255.0
        if self.is_tanh:
            img = img * 2 - 1
        return img

    def __call__(self, batch_size, random_flip = True):
        if self.pointer + batch_size > self.num_examples:
            rest_num_examples = self.num_examples - self.pointer
            path_rest_part = self.data[self.pointer:self.num_examples]
            labels_rest_part = self.labels[self.pointer:self.num_examples]
            self.shuffle_data()
            self.pointer = batch_size - rest_num_examples
            path_new_part = self.data[0:self.pointer]
            labels_new_part = self.labels[0:self.pointer]
            batch_path = np.concatenate((path_rest_part, path_new_part), axis=0)
            batch_data = [self.get_img(img_path) for img_path in batch_path]
            batch_data = np.array(batch_data)
            if random_flip:
                batch_data = self._random_flip_leftright(batch_data)
            return batch_data, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            start = self.pointer
            self.pointer += batch_size
            batch_path = self.data[start:self.pointer]
            batch_data = [self.get_img(img_path) for img_path in batch_path]
            batch_data = np.array(batch_data)
            if random_flip:
                batch_data = self._random_flip_leftright(batch_data)
            return batch_data, self.labels[start:self.pointer]

    def data2fig(self, samples, nr=4, nc=4):
        if self.is_tanh:
            samples = (samples + 1) / 2
        fig = plt.figure(figsize=(30, 6))
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.size, self.size, self.channel), cmap='Greys_r')
        return fig

class two_moon():
    def __init__(self):
        self.X_dim = 2 # for mlp
        self.z_dim = 4
        self.y_dim = 2
        self.data, self.labels = make_moons(n_samples=500000, noise=0.15, random_state=0)
        self.num_examples = len(self.data)

        self.pointer = 0

        self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(self.num_examples)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def __call__(self, batch_size, random_flip = True):
        if self.pointer + batch_size > self.num_examples:
            rest_num_examples = self.num_examples - self.pointer
            images_rest_part = self.data[self.pointer:self.num_examples]
            labels_rest_part = self.labels[self.pointer:self.num_examples]
            self.shuffle_data()
            self.pointer = batch_size - rest_num_examples
            images_new_part = self.data[0:self.pointer]
            labels_new_part = self.labels[0:self.pointer]
            batch_data = np.concatenate((images_rest_part, images_new_part), axis=0)
            return batch_data, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            start = self.pointer
            self.pointer += batch_size
            batch_data = self.data[start:self.pointer]
            return batch_data, self.labels[start:self.pointer]

    def data2fig(self, samples, labels, real_samples = None, real_labels = None):
        fig = plt.figure()
        plt.scatter(samples[labels == 0, 0], samples[labels == 0, 1], c='r', marker='o', label='class 0')
        plt.scatter(samples[labels == 1, 0], samples[labels == 1, 1], c='b', marker='s', label='class 1')

        if real_samples is not None:
            plt.scatter(real_samples[real_labels == 0, 0], real_samples[real_labels == 0, 1], c='r', marker='o', alpha = 0.5)
            plt.scatter(real_samples[real_labels == 1, 0], real_samples[real_labels == 1, 1], c='b', marker='s', alpha = 0.5)

        plt.xlim(samples[:, 0].min() - 1, samples[:, 0].max() + 1)
        plt.xlim(samples[:, 1].min() - 1, samples[:, 1].max() + 1)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='best')
        plt.tight_layout()
        return fig

def make_three_moons(n_samples=1500, shuffle=True, noise=None, random_state=None):
    """Make two interleaving half circles

    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.

    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.

    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    n_samples_one = n_samples // 3

    generator = check_random_state(random_state)

    one_circ_x = np.cos(np.linspace(0, np.pi, n_samples_one))
    one_circ_y = np.sin(np.linspace(0, np.pi, n_samples_one)) -.5
    two_circ_x = np.cos(np.linspace(0, np.pi, n_samples_one)) + 2.2
    two_circ_y = np.sin(np.linspace(0, np.pi, n_samples_one)) -.5
    three_circ_x = np.cos(np.linspace(0, np.pi, n_samples_one)) -2.2
    three_circ_y = np.sin(np.linspace(0, np.pi, n_samples_one)) -.5

    X = np.vstack((np.append(np.append(one_circ_x, two_circ_x), three_circ_x),
                   np.append(np.append(one_circ_y, two_circ_y), three_circ_y))).T
    y = np.hstack([np.zeros(n_samples_one, dtype=np.intp),
                   np.ones(n_samples_one, dtype=np.intp),
                   np.ones(n_samples_one, dtype=np.intp) * 2])

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y

class three_moon():
    def __init__(self):
        self.X_dim = 2 # for mlp
        self.z_dim = 2
        self.zp_dim = 2
        self.zc_dim = 2
        self.y_dim = 3
        self.data, self.labels = make_three_moons(n_samples=1800000, noise=0.15, random_state=0)
        self.num_examples = len(self.data)

        self.pointer = 0

        self.shuffle_data()

    def shuffle_data(self):
        indices = np.random.permutation(self.num_examples)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def __call__(self, batch_size, random_flip = True):
        if self.pointer + batch_size > self.num_examples:
            rest_num_examples = self.num_examples - self.pointer
            images_rest_part = self.data[self.pointer:self.num_examples]
            labels_rest_part = self.labels[self.pointer:self.num_examples]
            self.shuffle_data()
            self.pointer = batch_size - rest_num_examples
            images_new_part = self.data[0:self.pointer]
            labels_new_part = self.labels[0:self.pointer]
            batch_data = np.concatenate((images_rest_part, images_new_part), axis=0)
            return batch_data, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            start = self.pointer
            self.pointer += batch_size
            batch_data = self.data[start:self.pointer]
            return batch_data, self.labels[start:self.pointer]

    def data2fig(self, samples, labels, real_samples = None, real_labels = None):
        fig = plt.figure()

        if real_samples is not None:
            plt.scatter(real_samples[real_labels == 0, 0], real_samples[real_labels == 0, 1], c='black', alpha = 0.5)
            plt.scatter(real_samples[real_labels == 1, 0], real_samples[real_labels == 1, 1], c='black', alpha = 0.5)
            plt.scatter(real_samples[real_labels == 2, 0], real_samples[real_labels == 2, 1], c='black', alpha = 0.5)

        plt.scatter(samples[labels == 0, 0], samples[labels == 0, 1], c='r', label='class 0')
        plt.scatter(samples[labels == 1, 0], samples[labels == 1, 1], c='b', label='class 1')
        plt.scatter(samples[labels == 2, 0], samples[labels == 2, 1], c='g', label='class 2')

        plt.xlim(samples[:, 0].min() - 0.2, samples[:, 0].max() + 0.2)
        plt.ylim(samples[:, 1].min() - 0.2, samples[:, 1].max() + 0.2)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='best')
        plt.tight_layout()
        return fig








if __name__ == '__main__':
    sample_folder = 'Samples/three-moon-vae-adv-v2-gan-ngm3'
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    # data = SVHN(is_tanh=True)
    # data = three_moon()
    data = Cifar10()
    X_b, Y_b = data(200)
    # data.data2fig2jpg(X_b, Y_b)
    fig = data.data2fig(X_b, Y_b)
    plt.savefig('{}/{}.png'.format(sample_folder, 'real', bbox_inches='tight'))
    plt.close(fig)
    X_b, Y_b = data(900)
    fig = data.data2fig(X_b, Y_b)
    plt.savefig('{}/{}.png'.format(sample_folder, 'real_2', bbox_inches='tight'))
    plt.close(fig)