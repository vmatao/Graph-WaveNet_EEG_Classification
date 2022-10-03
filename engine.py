import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, gcn_bool,
                 addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = nn.CrossEntropyLoss()
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        loss = self.loss(output, real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        accuracy, dict_accuracy_df = util.accuracy(output, real_val)
        return loss.item(), accuracy, dict_accuracy_df

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        loss = self.loss(output, real_val)
        accuracy, dict_accuracy_df = util.accuracy(output, real_val)
        return loss.item(), accuracy, dict_accuracy_df
