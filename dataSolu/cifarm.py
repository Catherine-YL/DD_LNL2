from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from .utils_noise import download_url, check_integrity, noisify,noisify_instance

# for Synthetic Noise

class CIFAR10m(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.targets = []
        self.noise_type=noise_type
        self.noise_rate=noise_rate
        self.nb_classes=10
        idx_each_class_noisy = [[] for i in range(10)]
        
        if(self.noise_rate == 0.0):
            self.noise_type = 'clean'
        elif(self.noise_type  == 'clean'):
            self.noise_rate = 0.0
            
        if download:
           self.download()
           
        if not self._check_integrity():
           raise RuntimeError('Dataset not found or corrupted.' +
                              ' You can use download=True to download it')
        
        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            #if noise_type is not None:
            if noise_type !='clean':
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
                # self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels,noise_rate=noise_rate)
                print('over all noise rate is ', self.actual_noise_rate)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                #_train_labels=[i[0] for i in self.train_labels]
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
#                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
            else:
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.actual_noise_rate = 0
                print('over all noise rate is ', self.actual_noise_rate)
                self.train_labels=[i[0] for i in self.train_labels]
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_labels[i]].append(i)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            
        if self.train:
            if self.noise_type !='clean':
                self.targets = self.train_noisy_labels
            else:
                self.targets = self.train_labels
        else:
            self.targets = self.test_labels
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
                target_clean = self.train_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.train:
            if self.noise_type !='clean':
                return img, target, target_clean

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
               
    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100m(data.Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar100'
        self.targets = []
        self.noise_type=noise_type
        self.noise_rate=noise_rate
        self.nb_classes=100
        idx_each_class_noisy = [[] for i in range(100)]

        if(self.noise_rate == 0.0):
            self.noise_type = 'clean'
        elif(self.noise_type  == 'clean'):
            self.noise_rate = 0.0
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            if noise_type !='clean':
                # train_noisy_labels = self.load_label()
                # self.train_noisy_labels = train_noisy_labels.numpy().tolist()
                # print(f'noisy labels loaded from {self.noise_type}')
                # noisify train data
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
                # self.train_noisy_labels, self.actual_noise_rate = noisify_instance(self.train_data, self.train_labels,noise_rate=noise_rate)
                print('over all noise rate is ', self.actual_noise_rate)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                #_train_labels=[i[0] for i in self.train_labels]
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
#                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
            else:
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.actual_noise_rate = 0
                print('over all noise rate is ', self.actual_noise_rate)
                self.train_labels=[i[0] for i in self.train_labels]
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_labels[i]].append(i)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        if self.train:
            if self.noise_type !='clean':
                self.targets = self.train_noisy_labels
            else:
                self.targets = self.train_labels
        else:
            self.targets = self.test_labels

    def load_label(self):
            '''
            I adopt .pt rather .pth according to this discussion:
            https://github.com/pytorch/pytorch/issues/14864
            '''
            #NOTE presently only use for load manual training label
            # noise_label = torch.load(self.noise_type)
            noise_label = torch.load(f'noise_label/cifar-100/{self.noise_type}')
            if isinstance(noise_label, dict):
                if "clean_label_train" in noise_label.keys():
                    clean_label = noise_label['clean_label_train']
                    assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
                return noise_label['noise_label_train'] # % 10
            else:
                return noise_label  # % 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


