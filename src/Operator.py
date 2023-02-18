import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

class OperatorNetwork:

    def __init__(self, x_batch_size, mask_batch_size):
        self.batch_size = mask_batch_size * x_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = x_batch_size
        self.losses_per_sample = None
        self.tr_loss_history = []
        self.te_loss_history = []
        self.epoch_counter = 0

        self.lr = 0.001

    def create_dense_model(self, input_shape, dense_arch):

        self.model = nn.ModuleList([nn.Linear(input_shape[0] * 2, dense_arch[0])])
        self.model.append(nn.ReLU())
        self.model.append(nn.Dropout(0.5))

        for i in range(len(dense_arch) - 2):
            self.model.append(nn.Linear(dense_arch[i], dense_arch[i+1]))
            self.model.append(nn.ReLU())
            self.model.append(nn.Dropout(0.5))

        self.model.append(nn.Linear(dense_arch[-2], dense_arch[-1]))
        self.model.append(nn.Softmax())

        self.model.cuda()
        print(self.model)
        print("Object network model built:")

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def create_batch(self, x, masks, y):
        """
        x =     [[1,2],[3,4]]       -> [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]]
        masks = [[0,0],[1,0],[1,1]] -> [[0,0],[1,0],[1,1],[0,0],[1,0],[1,1]]
        y =     [1,3]               -> [1    ,1    ,1    ,3    ,3    ,3    ]
        """
        x_prim = np.repeat(x, len(masks), axis=0)
        y_prim = np.repeat(y, len(masks), axis=0)
        masks_prim = np.tile(masks, (len(x), 1))

        x_prim *= masks_prim  # MASKING

        return x_prim, masks_prim, y_prim

    def loss(self, outputs, targets):

        losses_per_sample = - torch.log(torch.sum(outputs * targets, 1))
        self.losses_per_sample = losses_per_sample.data.cpu().numpy()
        average_losses = torch.sum(losses_per_sample)/outputs.shape[0]

        return average_losses

    def get_per_mask_loss(self, used_target_shape=None):
        if used_target_shape is None:
            used_target_shape = (self.x_batch_size, self.mask_batch_size)
        losses = np.reshape(self.losses_per_sample, used_target_shape)

        losses = np.mean(losses, axis=0)
        return losses

    def my_parameters(self):
        return [{'params': self.model.parameters()}]

    def initialize(self):

        self.optimizer = optim.Adam(self.my_parameters(), lr=self.lr)

    def train_one(self, x, masks, y):

        x_prim, masks_prim, y_prim = self.create_batch(x, masks, y)

        inputs = Variable(torch.tensor(np.concatenate([masks_prim, x_prim * masks_prim], axis=1))).cuda()
        targets = Variable(torch.tensor(y_prim)).cuda()

        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        curr_loss = self.loss(outputs, targets)
        curr_loss.backward()
        self.optimizer.step()

        if self.epoch_counter % 500 == 0:
            print('Epoch:[{}]: operator training loss={:.5f}'.format(self.epoch_counter, curr_loss))

        self.tr_loss_history.append(curr_loss)
        self.epoch_counter += 1
        return x_prim, masks_prim, y_prim
