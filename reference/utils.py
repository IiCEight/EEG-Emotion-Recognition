import numpy as np
import scipy.io as scio
import torch
from torch.utils import data
from torch.utils.data import Dataset
import scipy.io


def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z = (X-mean) / (std+0.000000001)

    return z, mean, std


def load_seed4(path, session="all", feature="de_LDS", n_samples=100):
    """
       SEED IV
       A total number of 15 subjects participated the experiment. For each participant,
       3 sessions are performed on different days, and each session contains 24 trials.
       In one trial, the participant watch one of the film clips, while his(her) EEG
       signals and eye movements are collected with the 62-channel ESI NeuroScan System
       and SMI eye-tracking glasses.

       """

    # SESSION 1
    session1 = [
        "1_20160518",
        "2_20150915",
        "3_20150919",
        "4_20151111",
        "5_20160406",
        "6_20150507",
        "7_20150715",
        "8_20151103",
        "9_20151028",
        "10_20151014",
        "11_20150916",
        "12_20150725",
        "13_20151115",
        "14_20151205",
        "15_20150508"
    ]
    # SESSION 2
    session2 = [
        "1_20161125",
        "2_20150920",
        "3_20151018",
        "4_20151118",
        "5_20160413",
        "6_20150511",
        "7_20150717",
        "8_20151110",
        "9_20151119",
        "10_20151021",
        "11_20150921",
        "12_20150804",
        "13_20151125",
        "14_20151208",
        "15_20150514"
    ]
    # SESSION 3
    session3 = [
        "1_20161126",
        "2_20151012",
        "3_20151101",
        "4_20151123",
        "5_20160420",
        "6_20150512",
        "7_20150721",
        "8_20151117",
        "9_20151209",
        "10_20151023",
        "11_20151011",
        "12_20150807",
        "13_20161130",
        "14_20151215",
        "15_20150527"
    ]

    # select session
    if session == 1:
        x_session = session1
        y_session = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    elif session == 2:
        x_session = session2
        y_session = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    elif session == 3:
        x_session = session3
        y_session = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + str(session) + "/" + subj + ".mat", mat_dtype=True)
        for i in range(24):

            features = dataMat[feature + str(i + 1)]

            # swap frequency bands with epochs
            features = np.swapaxes(features, 0, 1)

            # select last samples
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]

            # set labels for each epoch
            labels = np.array([y_session[i]] * features.shape[0])

            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)
                Y = np.concatenate((Y, labels), axis=0)

        if samples_by_subject == 0:
            samples_by_subject = len(X)

    # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r + n]
        Y_subjects[subj] = Y[r:r + n]
        # increment range
        r += n

    return X_subjects, Y_subjects


class PairedData(object):
    def __init__(self, data_loader_src, data_loader_trg, max_dataset_size, num_domains_src):
        self.data_loader_src = data_loader_src
        self.data_loader_trg = data_loader_trg
        self.stop_src = [False]*num_domains_src
        self.stop_trg = False
        self.max_dataset_size = max_dataset_size
        self.num_domains_src = num_domains_src

    def __iter__(self):
        self.data_loader_src_iter = []
        for i in range(self.num_domains_src):
            self.stop_src[i] = False
            self.data_loader_src_iter.append(iter(self.data_loader_src[i]))

        self.stop_trg = False
        self.data_loader_trg_iter = iter(self.data_loader_trg)
        self.iter = 0
        return self

    def __next__(self):
        # initialize
        src_x = []
        src_y = []

        stop = True

        for i in range(self.num_domains_src):
            src_x.append(None)
            src_y.append(None)
            try:
                src_x[i], src_y[i] = next(self.data_loader_src_iter[i])
            except StopIteration:
                if src_x[i] is None or src_y[i] is None:
                    self.stop_src[i] = True
                    self.data_loader_src_iter[i] = iter(self.data_loader_src[i])
                    src_x[i], src_y[i] = next(self.data_loader_src_iter[i])

            if not self.stop_src[i]:
                stop = False

        trg_x, trg_y = None, None
        try:
            trg_x, trg_y = next(self.data_loader_trg_iter)
        except StopIteration:
            if trg_x is None or trg_y is None:
                self.stop_trg = True
                self.data_loader_trg_iter = iter(self.data_loader_trg)
                trg_x, trg_y = next(self.data_loader_trg_iter)

        if (stop and self.stop_trg) or self.iter > self.max_dataset_size:
            for i in range(self.num_domains_src):
                self.stop_src[i] = False
            self.stop_trg = False
            raise StopIteration()

        else:
            self.iter += 1
            data = {}
            # add source data
            for i in range(self.num_domains_src):
                data["Sx" + str(i + 1)] = src_x[i]
                data["Sy" + str(i + 1)] = src_y[i]
            # add target data
            data["Tx"] = trg_x
            data["Ty"] = trg_y

            return data

class UnalignedDataLoader():
    def initialize(self, num_domains, Sx, Sy, Tx, Ty, trg_subject, batch_size_src, batch_size_trg, drop_last_testing, shuffle_testing):

        # source domain
        self.dataset_src = []
        data_loader_src = []

        #####################################
        ###### MULTIPLES DOMINIOS ###########
        #####################################
        num_domains_src = num_domains - 1
        # print("[*] Target subject", trg_subject)
        # Store SOURCE DOMAINS
        for i in range(num_domains):
            if i != trg_subject:
                # obtain data from subject 's'
                x_tr = np.array(Sx[i])
                y_tr = np.array(Sy[i])

                # Standardize training data
                x_tr, m, std = z_score(x_tr)

                # print("Subject", str(i + 1), " Total:", len(y_tr), "  Partial:", a, b, c, " Minimum:", minimum, " # classes:", num_classes)
                dataset = Dataset(x_tr, y_tr)
                self.dataset_src.append(dataset)
                data_loader_src.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size_src, shuffle=True, drop_last=True))
        #################################################


        # Store TARGET DOMAIN
        dataset_target = Dataset(Tx, Ty)
        data_loader_trg = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size_trg, shuffle=shuffle_testing, drop_last=drop_last_testing)

        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_src, data_loader_trg, float("inf"), num_domains_src)
        self.num_domains_src = num_domains_src

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        maxim = -1
        for i in range(self.num_domains_src):
            m = len(self.dataset_src[i])
            if m > maxim:
                maxim = m

        return min(max(maxim, len(self.dataset_t)), float("inf"))

class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, data, label,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """

        img, target = self.data[index], self.labels[index]

        return img, target
    def __len__(self):
        return len(self.data)