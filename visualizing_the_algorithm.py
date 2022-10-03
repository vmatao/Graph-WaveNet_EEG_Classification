import time
import random

import matplotlib
import numpy
import winsound

import torch
from matplotlib import image as mpimg

import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import gif
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/BCI/testing', help='data path')
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
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
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
        torch.load(
            "garage/bci/exp50_62_before_or_after_7030_lr_0.0001_batch_128_both/_exp20220607155738_best_92.91.pth"))

    model.eval()

    print('model load successfully')

    scaler_cont = None
    dataloader, scaler = util.load_whole_exp(args.data, args.batch_size, '9', scaler_cont, args.batch_size,
                                             args.batch_size)
    accuracy = []
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    np.seterr(invalid='ignore')

    ims = []
    gif.options.matplotlib["dpi"] = 100
    fig, axs = plt.subplots(2, 2)
    iterstop = 1000
    event_location = 0
    helper = 0
    for iter, (x, y) in enumerate(dataloader['9_loader'].get_iterator()):
        helper += 1
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testx = nn.functional.pad(testx, (1, 0, 0, 0))
        testy = torch.Tensor(y).to(device)
        with torch.no_grad():
            preds = model(testx)
        metrics = util.metric_visual(preds, testy)
        test_accuracy_df = test_accuracy_df.append(metrics[1])
        eegs = testx.cpu().numpy()
        eegs = np.reshape(eegs, (22, 51))
        pred = preds.cpu().numpy()
        real = testy.cpu().numpy()
        c_real = np.argmax(real, axis=1)
        c_pred = np.argmax(pred, axis=1)
        accuracy.append(metrics[0])
        current_acc = np.round(np.divide(sum(accuracy), len(accuracy)), 2)
        to_print = 'Total accuracy: ' + str(current_acc) + "%," + \
                   " L hand accuracy: " + str(np.round(test_accuracy_df['0'].mean())) + "%, R hand accuracy:" + str(
            np.round(test_accuracy_df['1'].mean())) + \
                   "%, Legs accuracy: " + str(np.round(test_accuracy_df['2'].mean())) + "%, Tongue accuracy:" + \
                   str(np.round(test_accuracy_df['3'].mean())) + "%, No movement accuracy: " + \
                   str(np.round(test_accuracy_df['4'].mean())) + "%"
        eegs = np.delete(eegs, 0, axis=1)
        eegs = np.transpose(eegs)

        movement_dict = {0: "movement_imgs/left_hand", 1: "movement_imgs/right_hand", 2: "movement_imgs/legs",
                         3: "movement_imgs/tongue", 4: "movement_imgs/still"}

        movement_dict_titles = {0: "L hand", 1: "R hand", 2: "Legs",
                                3: "Tongue", 4: "No movement"}

        if c_real[0] != 4:
            img = mpimg.imread(movement_dict[c_real[0]] + "_green.png")
        else:
            img = mpimg.imread(movement_dict[c_real[0]] + ".png")

        matching = False
        if c_pred[0] == c_real[0]:
            im1 = axs[0, 0].imshow(img)
            im2 = axs[0, 1].imshow(img)

            matching = True
        else:
            im1 = axs[0, 0].imshow(img)
            if c_pred[0] != 4:
                img_wrong = mpimg.imread(movement_dict[c_pred[0]] + "_red.png")
            else:
                img_wrong = mpimg.imread(movement_dict[c_pred[0]] + ".png")
            im2 = axs[0, 1].imshow(img_wrong)

        text2 = axs[0, 0].text(x=0, y=-10, s="Real: " + movement_dict_titles[c_real[0]], fontsize=12)
        if matching:
            text3 = axs[0, 1].text(x=0, y=-10, s="CORRECT Prediction: " + movement_dict_titles[c_pred[0]], fontsize=12,
                                   fontdict={'color': "green"})
        else:
            text3 = axs[0, 1].text(x=0, y=-10, s="WRONG Prediction: " + movement_dict_titles[c_pred[0]], fontsize=12,
                                   fontdict={'color': "red"})

        axs[0, 1].get_xaxis().set_visible(False)
        axs[0, 1].get_yaxis().set_visible(False)
        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 0].get_yaxis().set_visible(False)

        eeg_displayed_spatially = eegs[0, :]
        coord = np.genfromtxt("coord.csv", delimiter=",")
        x = np.transpose(coord[:, 1])
        y = np.transpose(coord[:, 2])

        heatmap_data = np.zeros([6, 7])
        for i in range(0, len(x) - 1):
            a = int(x[i]) + 3
            b = int(y[i]) * -1
            heatmap_data[b, a] = eeg_displayed_spatially[i]
        masked_array = np.ma.masked_where(heatmap_data == heatmap_data[0, 0], heatmap_data)
        names = np.arange(0, 23)
        if helper == 13:
            event_location = 50
        lines = axs[1, 0].plot(eegs)
        l = 0
        for cl in colors.cnames:
            lines[l].set(color=cl)
            l += 1
            if l == len(lines):
                break
        if event_location > 0:
            x = [event_location, event_location, event_location, event_location, event_location]
            y = [-4, -2, 0, 2, 4]
            a, = axs[1, 0].plot(x, y, "black")
            lines.append(a)
            event_location -= 1
        if ((iter + 1) % 76) == 0:
            helper = 0

        axs[1, 0].set_ylabel("Normalized electrical signal (Hz)")
        axs[1, 0].set_xlabel("Time (4 ms)")
        axs[1, 0].set_xlabel("Time (4 ms)")
        axs[1, 0].set_ylim(-2.5, 3)
        axs[1, 0].set_facecolor('white')
        axs[1, 1].get_xaxis().set_visible(False)
        axs[1, 1].get_yaxis().set_visible(False)

        cmap = matplotlib.cm.viridis.copy()  # Can be any colormap that you want after the cm
        cmap.set_bad(color='white')

        im4 = axs[1, 1].imshow(masked_array, animated=True, cmap=cmap)
        text1 = axs[1, 1].text(x=-16, y=-9, s=to_print, fontsize=18)
        axs[1, 0].set_title("EEG over time. Vertical line is the movement intention")
        axs[1, 1].set_title("EEG spatial. Orientation: Face is up and ears left and right.")
        final_ims = [im1, im2, im4, text1, text2, text3]
        final_ims.extend(lines)
        ims.append(final_ims)

        if iter == iterstop:
            fig.colorbar(im4, ax=axs[1, 1])
            fig.set_figheight(12)
            fig.set_figwidth(20)
            ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                            repeat_delay=1000)
            ani.save("algo_viz_1000.gif")
            break


if __name__ == "__main__":
    main()

    duration = 500  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
