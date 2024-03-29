import time
import winsound

import torch

import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/BCI', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bci22.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly', type=bool, default=False, help='whether only adaptive adj')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')
parser.add_argument('--randomadj', type=bool, default=False, help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=50, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=22, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')

args = parser.parse_args()


def main():
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    nhid = args.nhid
    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj,
                  in_dim=args.in_dim, out_dim=args.seq_length, aptinit=adjinit, residual_channels=nhid,
                  dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
    model.to(device)
    model.load_state_dict(
        torch.load("garage/bci/exp50_62_before_or_after_7030_lr_0.0001_batch_128_both/_exp20220607155738_best_92.91.pth"))

    model.eval()

    print('model load successfully')

    outputs = []
    real = []
    scaler_cont = None
    time_per_50 = []
    for category in ['test50_62_0before_or_after_7030_all']:
        dataloader, scaler = util.load_whole_exp(args.data, args.batch_size, category, scaler_cont, args.batch_size,
                                                 args.batch_size)
        scaler_cont = scaler
        # scaler = dataloader['scaler']
        cat_pred = []
        cat_real = []
        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader[category + '_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testx = nn.functional.pad(testx, (1, 0, 0, 0))
            testy = torch.Tensor(y).to(device)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds.squeeze())
            real.append(testy.squeeze())
            cat_pred.append(preds.squeeze())
            cat_real.append(testy.squeeze())
        t2 = time.time()
        cat_real = torch.cat(cat_real, dim=0)
        cat_pred = torch.cat(cat_pred, dim=0)
        cat_pred_4 = []
        cat_real_4 = []
        zero = torch.tensor([0., 0., 0., 0., 1.])
        zero = torch.Tensor(zero).to(device)
        for elem, aelem in zip(cat_real, cat_pred):
            if torch.equal(elem, zero):
                continue
            cat_real_4.append(elem.squeeze())
            cat_pred_4.append(aelem.squeeze())
        cat_real_4 = torch.cat(cat_real_4, dim=0)
        size = cat_real_4.size(dim=0)
        cat_real_4 = torch.reshape(cat_real_4, (int(size / 5), 5))
        cat_pred_4 = torch.cat(cat_pred_4, dim=0)
        size = cat_pred_4.size(dim=0)
        cat_pred_4 = torch.reshape(cat_pred_4, (int(size / 5), 5))

        # for y in zero:
        #     b = b[b.eq(y).all(dim=1).logical_not()]
        metrics = util.metric(cat_pred, cat_real)
        metrics_4 = util.metric(cat_pred_4, cat_real_4)
        t3 = time.time()
        print("Time experiment " + category + " per 50 0.004s windows " + str((t2-t1)/cat_real.size(dim=0)))
        time_per_50.append((t2-t1)/cat_real.size(dim=0))
        print("Time experiment " + category + " per 50 0.004s windows with metrics calc " + str((t3 - t1) / cat_real.size(dim=0)))
        # acc, ev_dict, kap, f, p
        log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F1: {:.2f}, Auc: {:.2f}'
        print(log.format(metrics[0], metrics[2], metrics[5], metrics[6]))
        print(metrics[7])
        test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
        test_accuracy_df = test_accuracy_df.append(metrics[1])
        print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
              format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                     test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

        log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F1: {:.2f}, Auc: {:.2f}'
        print(log.format(metrics_4[0], metrics_4[2], metrics_4[5], metrics_4[6]))
        print(metrics_4[7])
        test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
        test_accuracy_df = test_accuracy_df.append(metrics_4[1])
        print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
              format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                     test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(real, dim=0)

    tot_pred_4 = []
    tot_real_4 = []
    zero = torch.tensor([0., 0., 0., 0., 1.])
    zero = torch.Tensor(zero).to(device)
    for elem, real_elem in zip(yhat, realy):
        if torch.equal(real_elem, zero):
            continue
        tot_real_4.append(real_elem)
        tot_pred_4.append(elem)
    tot_real_4 = torch.cat(tot_real_4, dim=0)
    size = tot_real_4.size(dim=0)
    tot_real_4 = torch.reshape(tot_real_4, (int(size / 5), 5))
    tot_pred_4 = torch.cat(tot_pred_4, dim=0)
    size = tot_pred_4.size(dim=0)
    tot_pred_4 = torch.reshape(tot_pred_4, (int(size / 5), 5))

    print( "Time average per 50 0.004s windows " + str(np.mean(time_per_50)))

    metrics = util.metric(yhat, realy)
    metrics_4 = util.metric(tot_pred_4, tot_real_4)
    log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F1: {:.2f}, Auc: {:.2f}'
    print(log.format(metrics[0], metrics[2], metrics[5], metrics[6]))
    print(metrics[7])
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    test_accuracy_df = test_accuracy_df.append(metrics[1])
    print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
          format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                 test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

    log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F1: {:.2f}, Auc: {:.2f}'
    print(log.format(metrics_4[0], metrics_4[2], metrics_4[5], metrics_4[6]))
    print(metrics_4[7])
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    test_accuracy_df = test_accuracy_df.append(metrics_4[1])
    print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
          format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                 test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp * (1 / np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb" + '.pdf')


if __name__ == "__main__":
    main()

    duration = 3000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
