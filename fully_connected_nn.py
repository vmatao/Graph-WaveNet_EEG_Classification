import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Check Device configuration
import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 1100
hidden_size = 500
num_classes = 5
num_epochs = 200
batch_size = 32
learning_rate = 0.001
test= True

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

dataloader = util.load_dataset('data/BCI', batch_size, batch_size, batch_size)
train_loader = dataloader['train_loader']
test_loader = dataloader['test_loader']

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
if not test:
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = train_loader.size
    for epoch in range(num_epochs):
        for i, (pred, labels) in enumerate(train_loader.get_iterator()):
            # Move tensors to the configured device
            pred = torch.Tensor(pred).to(device)
            pred = pred.transpose(1, 3)
            pred = torch.flatten(pred,2)
            pred = torch.flatten(pred, 1)
            labels = torch.Tensor(labels).to(device)

            # Forward pass
            outputs = model(pred)
            loss = criterion(outputs, labels)

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        classes = ('0', '1', '2', '3', '4')
        dict_acc_ev = {}
        for pred, labels in test_loader.get_iterator():
            pred = torch.Tensor(pred).to(device)
            pred = pred.transpose(1, 3)
            pred = torch.flatten(pred,2)
            pred = torch.flatten(pred, 1)
            labels = torch.Tensor(labels).to(device)
            outputs = model(pred)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_val = torch.max(labels.data, 1)
            total += labels_val.size(0)
            correct += (predicted == labels_val).sum().item()

            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            # again no gradients needed
            _, predictions = torch.max(outputs, 1)
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



        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        print(df_class_accuracy.head())

    # Save the model checkpoint
    torch.save(model.state_dict(), 'garage/bci/exp_fnn_50_62_70_30_20_epoch/model2.pth')
else:
    model.load_state_dict(
        torch.load("garage/bci/exp_fnn_50_62_70_30_20_epoch/model2.pth"))

    print('model load successfully')

    outputs = []
    real = []
    # realy = torch.Tensor(dataloader['y_1']).to(device)
    # realy = realy.transpose(1,3)[:,0,:,:]
    scaler_cont = None
    time_per_50 = []
    for category in ['1', '2', '3', '5', '6', '7', '8', '9']:
        dataloader, scaler = util.load_whole_exp('data/BCI/testing', 32, category, scaler_cont, 32,
                                                 32)
        scaler_cont = scaler
        # scaler = dataloader['scaler']
        cat_pred = []
        cat_real = []
        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader[category + '_loader'].get_iterator()):

            # Forward pass
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testx = torch.flatten(testx, 2)
            testx = torch.flatten(testx, 1)
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
        print("Time experiment " + category + " per 50 0.004s windows " + str((t2 - t1) / cat_real.size(dim=0)))
        time_per_50.append((t2 - t1) / cat_real.size(dim=0))
        print("Time experiment " + category + " per 50 0.004s windows with metrics calc " + str(
            (t3 - t1) / cat_real.size(dim=0)))
        # acc, ev_dict, kap, f, p
        log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F-test: {:.2f}, P value: {:.2f}'
        print(log.format(metrics[0], metrics[2], metrics[3], metrics[4]))
        test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
        test_accuracy_df = test_accuracy_df.append(metrics[1])
        print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
              format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                     test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

        log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F-test: {:.2f}, P value: {:.2f}'
        print(log.format(metrics_4[0], metrics_4[2], metrics_4[3], metrics_4[4]))
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

    print("Time average per 50 0.004s windows " + str(np.mean(time_per_50)))

    metrics = util.metric(yhat, realy)
    metrics_4 = util.metric(tot_pred_4, tot_real_4)
    log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F-test: {:.2f}, P value: {:.2f}'
    print(log.format(metrics[0], metrics[2], metrics[3], metrics[4]))
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    test_accuracy_df = test_accuracy_df.append(metrics[1])
    print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
          format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                 test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))

    log = 'Test acc: {:.2f}, Kappa Value: {:.2f}, F-test: {:.2f}, P value: {:.2f}'
    print(log.format(metrics_4[0], metrics_4[2], metrics_4[3], metrics_4[4]))
    test_accuracy_df = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    test_accuracy_df = test_accuracy_df.append(metrics_4[1])
    print('0: {:.2f}%, 1: {:.2f}%, 2: {:.2f}%, 3: {:.2f}%, 4: {:.2f}%'.
          format(test_accuracy_df['0'].mean(), test_accuracy_df['1'].mean(),
                 test_accuracy_df['2'].mean(), test_accuracy_df['3'].mean(), test_accuracy_df['4'].mean()))