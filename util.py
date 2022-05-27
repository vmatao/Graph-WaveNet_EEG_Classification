import pickle
import numpy as np
import os

import pandas as pd
import scipy
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
import scipy.stats as ss
from sklearn.metrics import f1_score
from sklearn import metrics


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        # new_df = pd.DataFrame(columns=["idx", "event"])
        # for i in range(0, len(permutation)):
        #     if i in df.idx.values:
        #         a = permutation[i]
        #         b = df.loc[df['idx'] == i].iat[0, 1]
        #         new_df.loc[len(new_df.index)] = [a, b]
        # df = df.set_index("idx")
        # df = df.sort_index()
        #
        # new_df = new_df.set_index("idx")
        # new_df = new_df.sort_index()
        # d = {7: [1, 0, 0, 0], 8: [0, 1, 0, 0], 9: [0, 0, 1, 0], 10: [0, 0, 0, 1]}
        # df.event = df.event.map(d)
        # df.to_pickle("rest.pkl")
        # try:
        #     new_df.event = new_df.event.map(d)
        # except:
        #     print("new_df.event = new_df.event.map(d)")
        # self.shuffled_df = new_df.copy()

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]

                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    df1231 = pd.DataFrame(adj_mx[0])
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'test']:
        # np_load_old = np.load
        # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        # cat_data = np.load(os.path.join(dataset_dir, category + 'ww.npz'))
        cat_data = np.load(os.path.join(dataset_dir, category + '50_62_0before_or_after_7030.npz'))
        # np.load = np_load_old
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    # data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def load_whole_exp(dataset_dir, batch_size, category, scaler_cont, valid_batch_size=None, test_batch_size=None):
    data = {}
    category = str(category)
    # np_load_old = np.load
    # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # cat_data = np.load(os.path.join(dataset_dir, category + 'ww.npz'))
    cat_data = np.load(os.path.join(dataset_dir, category + 'whole_exp_testing.npz'))
    # np.load = np_load_old
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
    if scaler_cont is None:
        scaler = StandardScaler(mean=data['x_1'][..., 0].mean(), std=data['x_1'][..., 0].std())
    else:
        scaler = scaler_cont
    # # Data format
    # for category in ['1', '2', '3', '5', '6', '7', '8', '9']:
    # TODO remove scaler for label 0 - got 70% accuracy without the scaling
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data[category + '_loader'] = DataLoader(data['x_' + category], data['y_' + category], batch_size)
    # data['2_loader'] = DataLoader(data['x_2'], data['y_2'], valid_batch_size)
    # data['3_loader'] = DataLoader(data['x_3'], data['y_3'], test_batch_size)
    # data['5_loader'] = DataLoader(data['x_5'], data['y_5'], batch_size)
    # data['6_loader'] = DataLoader(data['x_6'], data['y_6'], valid_batch_size)
    # data['7_loader'] = DataLoader(data['x_7'], data['y_7'], test_batch_size)
    # data['8_loader'] = DataLoader(data['x_8'], data['y_8'], batch_size)
    # data['9_loader'] = DataLoader(data['x_9'], data['y_9'], valid_batch_size)
    # data['scaler'] = scaler
    print(category)
    return data, scaler


# def masked_mse(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = (preds - labels) ** 2
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)
#
#
# def masked_rmse(preds, labels, null_val=np.nan):
#     return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
#
#
# def masked_mae(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)
#
#
# def masked_mape(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels != null_val)
#     mask = mask.float()
#     mask /= torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds - labels) / labels
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)


def accuracy(preds, labels):
    dict_acc_ev = {}
    # , '4'
    classes = ('0', '1', '2', '3', '4')

    correct = 0
    total = 0
    result = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(preds.data, 1)
        _, labels_val = torch.max(labels.data, 1)
        # if torch.max(_) != 0:
        total += labels.size(0)
        correct += (predicted == labels_val).sum().item()
    # if total != 0:
    result = 100 * correct / total

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        _, predictions = torch.max(preds, 1)
        _, labels_val = torch.max(labels, 1)
        # if torch.max(_) != 0:
        # collect the correct predictions for each class
        for label, prediction in zip(labels_val, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            dict_acc_ev[classname] = accuracy
            # print(total_pred[classname])
            # if accuracy > 0:
            #     print(classname + " "+str(accuracy))

    df_class_accuracy = pd.DataFrame.from_dict(dict_acc_ev, orient='index')
    df_class_accuracy = df_class_accuracy.T
    return result, df_class_accuracy


# TODO confusion matrix, AUC-ROC


def metric(pred, real):
    # mae = masked_mae(pred, real, 0.0).item()
    # mape = masked_mape(pred,real,0.0).item()
    # rmse = masked_rmse(pred,real,0.0).item()
    acc, ev_dict = accuracy(pred, real)
    pred = pred.cpu().numpy()
    real = real.cpu().numpy()
    try:
        auc = metrics.roc_auc_score(real, pred, multi_class='ovr')
        c_real = np.argmax(real, axis=1)
        c_pred = np.argmax(pred, axis=1)
        kap = cohen_kappa_score(c_pred, c_real)
        f1 = f1_score(c_pred, c_real, average="weighted")
        c_matrix = metrics.confusion_matrix(c_real, c_pred)
    except:
        pred = np.delete(pred, 4, 1)
        real = np.delete(real, 4, 1)
        c_real = np.argmax(real, axis=1)
        c_pred = np.argmax(pred, axis=1)
        kap = cohen_kappa_score(c_pred, c_real)
        f1 = f1_score(c_pred, c_real, average="weighted")
        c_matrix = metrics.confusion_matrix(c_real, c_pred)
        auc = metrics.roc_auc_score(real, pred, multi_class='ovr')

    # print confusion matrix

    return acc, ev_dict, kap, 0, 0, f1, auc, c_matrix
