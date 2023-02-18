import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

class SelectorNetwork:
    def __init__(self, mask_batch_size):
        self.batch_size = mask_batch_size
        self.mask_batch_size = mask_batch_size
        self.tr_loss_history = []
        self.te_loss_history = []
        self.y_pred_std_history = []
        self.y_true_std_history = []
        self.epoch_counter = 0
        self.data_masks = None
        self.data_targets = None
        self.best_performing_mask = None
        self.sample_weights = None

        self.predict_loss = None

        self.lr = 0.001

    def create_dense_model(self, input_shape, dense_arch):
        print('dense_arch', dense_arch)

        if len(dense_arch) == 1:
            self.model = nn.ModuleList([nn.Linear(input_shape[0], dense_arch[0])])
        else:
            self.model = nn.ModuleList([nn.Linear(input_shape[0], dense_arch[0])])
            self.model.append(nn.Sigmoid())
            for i in range(len(dense_arch) - 2):
                self.model.append(nn.Linear(dense_arch[i], dense_arch[i+1]))
                self.model.append(nn.Sigmoid())

            self.model.append(nn.Linear(dense_arch[-2], dense_arch[-1]))

        self.model.cuda()
        print(self.model)
        print("Object network model built:")

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def loss(self, outputs, targets, apply_weights):

        # outputs: output of the selector
        # targets: the averaged cross entropy

        if apply_weights == False:
            predict_loss = torch.mean(torch.abs(outputs - targets.reshape(-1, 1)))
        else:
            predict_loss = torch.mean(torch.abs(outputs - targets.reshape(-1, 1)) * torch.tensor(self.sample_weights).cuda())

        loss = predict_loss
        self.predict_loss = predict_loss

        return loss

    def my_parameters(self):
        return [{'params': self.model.parameters()}]

    def initialize(self):
        self.optimizer = optim.Adam(self.my_parameters(), lr=self.lr)

    def train_one(self, epoch_number, apply_weights):  # train on data in object memory

        inputs = Variable(torch.tensor(self.data_masks)).cuda()
        targets = Variable(torch.tensor(self.data_targets)).cuda()

        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        curr_loss = self.loss(outputs, targets, apply_weights)
        curr_loss.backward()
        self.optimizer.step()

        if self.epoch_counter % 500 == 1:
            print('Selector training loss: %.5f' % (curr_loss.item()))

        self.best_performing_mask = self.data_masks[np.argmin(self.data_targets, axis=0)]
        self.tr_loss_history.append(curr_loss)
        self.epoch_counter = epoch_number
        self.data_masks = None
        self.data_targets = None

    def append_data(self, x, y):
        if self.data_masks is None:
            self.data_masks = x
            self.data_targets = y
        else:
            self.data_masks = np.concatenate([self.data_masks, x], axis=0)
            self.data_targets = np.concatenate([self.data_targets, y], axis=0)

