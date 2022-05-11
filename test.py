import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        torch.load("garage/bci/exp50_50_100_0rand_split_suffle_8020/_exp20220428123034_best_69.9.pth"))

    model.eval()

    print('model load successfully')

    outputs = []
    real = []
    # realy = torch.Tensor(dataloader['y_1']).to(device)
    # realy = realy.transpose(1,3)[:,0,:,:]
    scaler_cont = None
    for category in ['1', '2', '3', '5', '6', '7', '8', '9']:
        dataloader, scaler = util.load_whole_exp(args.data, args.batch_size, category, scaler_cont, args.batch_size,
                                                 args.batch_size)
        scaler_cont = scaler
        # scaler = dataloader['scaler']
        a = []
        b = []
        for iter, (x, y) in enumerate(dataloader[category + '_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testx = nn.functional.pad(testx, (1, 0, 0, 0))
            testy = torch.Tensor(y).to(device)
            with torch.no_grad():
                preds = model(testx)
            outputs.append(preds.squeeze())
            real.append(testy.squeeze())
            a.append(preds.squeeze())
            b.append(testy.squeeze())
        a = torch.cat(a, dim=0)
        b = torch.cat(b, dim=0)
        metrics = util.metric(a, b)
        # acc, ev_dict, kap, f, p
        log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F-test: {:.2f}, P value: {:.2f}'
        print(log.format(metrics[0], metrics[2], metrics[3], metrics[4]))
        test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
        test_accuracy_df = test_accuracy_df.append(metrics[1])
        print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
              format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                     test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(real, dim=0)

    # amae = []
    # amape = []
    # armse = []
    # for i in range(12):
    #     pred = scaler.inverse_transform(yhat[:,:,i])
    #     real = realy[:,:,i]
    metrics = util.metric(yhat, realy)
    log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F-test: {:.2f}, P value: {:.2f}'
    print(log.format(metrics[0], metrics[2], metrics[3], metrics[4]))
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    test_accuracy_df = test_accuracy_df.append(metrics[1])
    print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
          format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                 test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])
    #
    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))

    if args.plotheatmap == "True":
        adp = F.softmax(F.relu(torch.mm(model.nodevec1, model.nodevec2)), dim=1)
        device = torch.device('cpu')
        adp.to(device)
        adp = adp.cpu().detach().numpy()
        adp = adp * (1 / np.max(adp))
        df = pd.DataFrame(adp)
        sns.heatmap(df, cmap="RdYlBu")
        plt.savefig("./emb" + '.pdf')

    # y12 = realy[:,99,11].cpu().detach().numpy()
    # yhat12 = scaler.inverse_transform(yhat[:,99,11]).cpu().detach().numpy()
    #
    # y3 = realy[:,99,2].cpu().detach().numpy()
    # yhat3 = scaler.inverse_transform(yhat[:,99,2]).cpu().detach().numpy()
    #
    # df2 = pd.DataFrame({'real12':y12,'pred12':yhat12, 'real3': y3, 'pred3':yhat3})
    # df2.to_csv('./wave.csv',index=False)


if __name__ == "__main__":
    main()
