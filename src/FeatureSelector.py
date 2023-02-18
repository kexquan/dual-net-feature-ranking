import numpy as np
from .MaskOptimizer import MaskOptimizer
from .Operator import OperatorNetwork
from .Selector import SelectorNetwork

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch

def progressbar(it, prefix="", size=60):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print("\r%s[%s%s] %i/%i" % (prefix, "#" * x, "." * (size - x), j, count), end=" ")

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print()

class FeatureSelector():
    def __init__(self, data_shape, unmasked_data_size, data_batch_size, mask_batch_size,
                 epoch_on_which_selector_trained=8):
        self.data_shape = data_shape
        self.data_size = np.zeros(data_shape).size
        self.unmasked_data_size = unmasked_data_size
        self.data_batch_size = data_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = mask_batch_size * data_batch_size
        self.epoch_on_which_selector_trained = epoch_on_which_selector_trained

    def create_dense_operator(self, arch):
        self.operator = OperatorNetwork(self.data_batch_size, self.mask_batch_size)
        print("Creating operator model")
        self.operator.create_dense_model(self.data_shape, arch)
        print("Created operator")

    def create_dense_selector(self, arch):
        self.selector = SelectorNetwork(self.mask_batch_size)
        self.selector.create_dense_model(self.data_shape, arch)

    def create_mask_optimizer(self, epoch_condition=5000, perturbation_size=2):
        self.mopt = MaskOptimizer(self.mask_batch_size, self.data_shape, self.unmasked_data_size,
                                  epoch_condition=epoch_condition, perturbation_size=perturbation_size)

        self.selector.sample_weights = self.mopt.get_mask_weights(self.epoch_on_which_selector_trained)

    def train_networks_on_data(self, x_tr, y_tr, number_of_batches, args):

        self.operator.initialize()
        self.selector.initialize()

        x_tr = torch.tensor(x_tr)
        y_tr = torch.tensor(y_tr)
        train_data = TensorDataset(x_tr, y_tr)

        if args.class_weighted:

            y_tr_temp = np.array(np.argmax(y_tr, axis=1))

            class_sample_count = np.array([len(np.where(y_tr_temp == t)[0]) for t in np.unique(y_tr_temp)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_tr_temp])
            samples_weight = torch.from_numpy(samples_weight)
            train_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        else:
            train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.data_batch_size, shuffle=False,
                                      worker_init_fn=np.random.seed(args.seed), drop_last=True)


        train_generator = iter(train_dataloader)

        for i in progressbar(range(number_of_batches), "Batch:", 50):

            mopt_condition = self.mopt.check_condiditon()
            try:
                x, y = next(train_generator)
            except StopIteration:
                train_generator = iter(train_dataloader)
                x, y = next(train_generator)

            selector_train_condition = ((self.operator.epoch_counter % self.epoch_on_which_selector_trained) == 0)
            m = self.mopt.get_new_mask_batch(self.selector, self.selector.best_performing_mask,
                                             gen_new_opt_mask=selector_train_condition)

            self.operator.train_one(x, m, y)
            losses = self.operator.get_per_mask_loss()
            self.selector.append_data(m, losses)
            if (selector_train_condition):
                
                self.selector.train_one(self.operator.epoch_counter, mopt_condition)

    def get_importances(self,):
        grad_used_opt = MaskOptimizer.get_feature_importance_vector(self.selector)
        importances = -grad_used_opt
        return importances
