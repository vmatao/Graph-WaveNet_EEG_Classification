import pickle
import numpy as np
import os

import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import tensorflow as tf

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

    def shuffle(self, df):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        new_df = pd.DataFrame(columns=["idx", "event"])
        for i in range(0, len(permutation)):
            if i in df.idx.values:
                a = permutation[i]
                b = df.loc[df['idx'] == i].iat[0, 1]
                new_df.loc[len(new_df.index)] = [a, b]
        return new_df

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
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
    for category in ['train', 'val', 'test']:
        # np_load_old = np.load
        # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        # cat_data = np.load(os.path.join(dataset_dir, category + 'ww.npz'))
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        # np.load = np_load_old
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def accuracy(preds, labels):
    # preds[preds > 0.4] = 1
    # preds[preds <= 0.4] = 0
    #
    # zero = preds[:,0]
    # one = preds[:,1]
    # two = preds[:,2]
    # three = preds[:,3]
    # four = preds[:,4]
    # zero_l = labels[:,0]
    # one_l = labels[:,1]
    # two_l = labels[:,2]
    # three_l = labels[:,3]
    # four_l = labels[:,4]
    #
    # zero_l = zero_l.detach().clone()
    # zero_l[zero_l == 0] = 2
    # one_l = one_l.detach().clone()
    # one_l[one_l == 0] = 2
    # two_l = two_l.detach().clone()
    # two_l[two_l == 0] = 2
    # three_l = three_l.detach().clone()
    # three_l[three_l == 0] = 2
    # four_l = four_l.detach().clone()
    # four_l[four_l == 0] = 2
    #
    # count_zero = torch.sum(torch.eq(zero, zero_l).int()).item()
    # count_one = torch.sum(torch.eq(one, one_l).int()).item()
    # count_two = torch.sum(torch.eq(two, two_l).int()).item()
    # count_three = torch.sum(torch.eq(three, three_l).int()).item()
    # count_four = torch.sum(torch.eq(four, four_l).int()).item()
    #
    # denom_zero_l = zero_l[zero_l == 1].size(dim=0)
    # denom_one_l = one_l[one_l == 1].size(dim=0)
    # denom_two_l = two_l[two_l == 1].size(dim=0)
    # denom_three_l = three_l[three_l == 1].size(dim=0)
    # denom_four_l = four_l[four_l == 1].size(dim=0)
    #
    #
    # dict_acc_ev = {}
    # if denom_zero_l!=0:
    #     acc_zero = count_zero / denom_zero_l
    #     dict_acc_ev["7"] = acc_zero
    # if denom_one_l != 0:
    #     acc_one = count_one / denom_one_l
    #     dict_acc_ev["8"] = acc_one
    # if denom_two_l != 0:
    #     acc_two = count_two / denom_two_l
    #     dict_acc_ev["9"] = acc_two
    # if denom_three_l != 0:
    #     acc_three = count_three / denom_three_l
    #     dict_acc_ev["10"] = acc_three
    # if denom_four_l != 0:
    #     acc_four = count_four / denom_four_l
    #     dict_acc_ev["0"] = acc_four
    #
    # events_pred = torch.max(preds, 1).indices
    # labels_pred = torch.max(labels, 1).indices
    # correct_preds = torch.eq(events_pred, labels_pred)
    # as_ints = correct_preds.int()
    # count = torch.sum(as_ints,dim=0)
    # a = as_ints.size(dim=0)
    # result = count.item() / a
    dict_acc_ev = {}
    # , '4'
    classes = ('0', '1', '2', '3')

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
            dict_acc_ev[classname]=accuracy
            # print(total_pred[classname])
            # if accuracy > 0:
            #     print(classname + " "+str(accuracy))
    return result, dict_acc_ev


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    # mape = masked_mape(pred,real,0.0).item()
    # rmse = masked_rmse(pred,real,0.0).item()
    acc, ev_dict = accuracy(pred, real)
    return mae, acc, ev_dict
