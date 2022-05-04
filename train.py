from datetime import datetime
import os
import pickle

import pandas as pd
import torch
import numpy as np
import argparse
import time
import torch.nn as nn
from simple_model import OurNeuralNetwork

import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/BCI', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx_bci22.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
# parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
# parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
# parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
# parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
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
parser.add_argument('--epochs', type=int, default=20, help='')
parser.add_argument('--print_every', type=int, default=1500, help='')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--save', type=str, default='./garage/bci', help='save path')
parser.add_argument('--expid', type=int, default=datetime.now().strftime('%Y%m%d%H%M%S'), help='experiment id')
args = parser.parse_args()


def main():
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    df1231 = pd.DataFrame(adj_mx[0])
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)
    if os.path.exists(args.save + "/exp"+ str(args.expid)+ "/"):
        reply = str(input(f'{args.save + "/exp"+ str(args.expid)+ "/"} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.save + "/exp"+ str(args.expid)+ "/")
    print("start training...", flush=True)
    his_loss = []
    his_acc = []
    val_time = []
    train_time = []
    save_iter = 0
    for i in range(1, args.epochs + 1):
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        # train_mape = []
        # train_rmse = []
        train_acc = []
        # train_acc_dict = []
        tain_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            # with open("real_val", "rb") as fp:  # Unpickling
            #     batch_real_val = pickle.load(fp)
            # batch_real_val = dataloader['train_loader'].get_real_val()
            trainy = torch.Tensor(y).to(device)
            # trainy = trainy.transpose(1, 3)
            # a=trainy[:,0,:,:]
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            # train_mape.append(metrics[1])
            # train_rmse.append(metrics[2])
            # if metrics[1] >= 0:
            train_acc.append(metrics[1])
            tain_accuracy_df = tain_accuracy_df.append(metrics[2])
            # if len(metrics[2].keys()) > 1:
            #     print(metrics[2])
            #     print(metrics[1])
            if iter % args.print_every == 0:
                # log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                # print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Accuracy: {:.2f}'
                # if len(train_acc) > 0:
                print(log.format(iter, train_loss[-1], train_acc[-1]), flush=True)
                # print('Rolling avg: 0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%,3: {:.2f}%'.
                #       format(tain_accuracy_df['0'].mean(), tain_accuracy_df['1'].mean(),
                #              tain_accuracy_df['2'].mean(), tain_accuracy_df['3'].mean()))
                # if len(train_acc) > 0:
                #     print(log.format(iter, train_loss[-1], -1), flush=True)
            save_iter = iter
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        # valid_mape = []
        # valid_rmse = []
        valid_acc = []
        # valid_acc_dict = []
        valid_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])

        save_iter1 = 0
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            # testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            # valid_mape.append(metrics[1])
            # valid_rmse.append(metrics[2])
            # if metrics[1] >= 0:
            valid_acc.append(metrics[1])
            valid_accuracy_df = valid_accuracy_df.append(metrics[2])
            # if len(metrics[2].keys()) > 1:
            #     print(metrics[2])
            # save_iter1 = iter * args.batch_size + save_iter
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))

        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        # mtrain_mape = np.mean(train_mape)
        # mtrain_rmse = np.mean(train_rmse)
        mtrain_acc = np.mean(train_acc)

        mvalid_loss = np.mean(valid_loss)
        # mvalid_mape = np.mean(valid_mape)
        # mvalid_rmse = np.mean(valid_rmse)
        mvalid_acc = np.mean(valid_acc)
        his_loss.append(mvalid_loss)
        his_acc.append(mvalid_acc)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train avg accuracy: {:.2f}%, Test Loss: {:.4f}, Test accuracy: {:.2f}%, ' \
              'Training Time: {:.4f}/epoch '
        # print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
        #       flush=True)
        print(log.format(i, mtrain_loss, mtrain_acc, mvalid_loss, mvalid_acc, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "/exp"+ str(args.expid)+ "/"+ "_epoch_" + str(i) + "_" + str(round(mvalid_acc, 2)) + ".pth")
        print("Train_dict: ")
        print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}% ,3: {:.2f}%, 4: {:.2f}%'.
              format(tain_accuracy_df['0'].mean(), tain_accuracy_df['1'].mean(),
                     tain_accuracy_df['2'].mean(), tain_accuracy_df['3'].mean(), tain_accuracy_df['4'].mean()))
        print("Test_dict: ")
        print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}% '.
              format(valid_accuracy_df['0'].mean(), valid_accuracy_df['1'].mean(),
                     valid_accuracy_df['2'].mean(), valid_accuracy_df['3'].mean(), valid_accuracy_df['4'].mean()))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmax(his_acc)
    engine.model.load_state_dict(
        torch.load(args.save + "/exp"+ str(args.expid)+ "/"+ "_epoch_" + str(bestid + 1) + "_" + str(round(his_acc[bestid], 2)) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy.transpose(1, 3)[:, 0, :, :]
    # batch_real_val = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            # .transpose(1, 3)
            testx = nn.functional.pad(testx, (1, 0, 0, 0))
            preds = engine.model(testx)
            # preds = engine.model.rest_of_operations(preds, scaler)
        outputs.append(preds.squeeze())
    # realy = torch.Tensor(y).to(device)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    amae = []
    # amape = []
    # armse = []
    acc = []
    # acc_dict = []
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    # for i in range(12):
    # pred = scaler.inverse_transform(yhat[:, :, i])
    # real = realy[:, :, i]
    # metrics = util.metric(yhat[:, :, i], realy)
    metrics = util.metric(yhat, realy)
    # log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test Acc: {:.4f}'
    #     print(log.format(i + 1, metrics[0], metrics[1]))
    log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test Acc: {:.4f}'
    print(log.format(0, metrics[0], metrics[1]))
    amae.append(metrics[0])
    # amape.append(metrics[1])
    # if metrics[1] >= 0:
    acc.append(metrics[1])
    test_accuracy_df = test_accuracy_df.append(metrics[2])

    log = 'On average over 1 horizons, Test MAE: {:.4f}, Test Acc: {:.4f}'
    print(log.format(np.mean(amae), np.mean(acc)))
    print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
          format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                 test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))
    torch.save(engine.model.state_dict(),
               args.save + "/exp"+ str(args.expid)+ "/"+ "_exp" + str(args.expid) + "_best_" + str(round(his_acc[bestid], 2)) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
