import numpy as np
import torch

class MaskOptimizer:
    def __init__(self, mask_batch_size, data_shape, unmasked_data_size,perturbation_size,
                 frac_of_rand_masks=0.5, epoch_condition=1000):
        self.data_shape = data_shape
        self.unmasked_data_size = unmasked_data_size
        self.data_size = np.zeros(data_shape).size
        self.mask_history = []
        self.raw_mask_history = []
        self.loss_history = []
        self.epoch_counter = 0
        self.mask_batch_size = mask_batch_size
        self.frac_of_rand_masks = frac_of_rand_masks
        self.epoch_condition = epoch_condition
        self.perturbation_size = perturbation_size
        self.max_optimization_iters = 5
        self.step_count_history = []

    def get_feature_importance_vector(model,):

        layers = model.model
        dense_layers_weights = []
        dense_layers_bias = []
        for i in range(len(layers)):  # dense_layer
            dense_layer = layers[i]
            if dense_layer._get_name() == 'Linear':
                dense_layers_weights.append(dense_layer.weight.T)
                dense_layers_bias.append(dense_layer.bias)

        if len(dense_layers_weights) == 1:
            temp = dense_layers_weights[-1] + dense_layers_bias[-1]
            c = temp.detach().cpu().numpy().squeeze()
        else:
            temp = torch.sigmoid(dense_layers_weights[0] + dense_layers_bias[0])
            for i in range(1, len(dense_layers_weights) - 1):
                temp = torch.sigmoid(torch.matmul(temp, dense_layers_weights[i]) + dense_layers_bias[i])
            temp = torch.matmul(temp, dense_layers_weights[-1]) + dense_layers_bias[-1]
            c = temp.detach().cpu().numpy().squeeze()

        return c

    def new_get_mask(grads, unmasked_size, mask_size):

        m_opt = np.zeros(shape=mask_size)
        top_arg = np.argpartition(grads, -unmasked_size)[-unmasked_size:]
        m_opt[top_arg] = 1
        return m_opt

    def new_get_m_opt(model, unmasked_size):

        c = MaskOptimizer.get_feature_importance_vector(model)
        c = np.negative(c)
        m_opt = MaskOptimizer.new_get_mask(c, unmasked_size, model.model[0].in_features)
        return m_opt

    def get_opt_mask(self, unmasked_size, model):
        m_opt = MaskOptimizer.new_get_m_opt(model, unmasked_size)
        return m_opt

    def check_condiditon(self):
        if (self.epoch_counter >= self.epoch_condition):
            return True
        else:
            return False

    def get_random_masks(self):
        masks_zero = np.zeros(shape=(self.mask_batch_size, self.data_size - self.unmasked_data_size))
        masks_one = np.ones(shape=(self.mask_batch_size, self.unmasked_data_size))
        masks = np.concatenate([masks_zero, masks_one], axis=1)
        masks_permuted = np.apply_along_axis(np.random.permutation, 1, masks)
        return masks_permuted

    def get_perturbed_masks(mask, n_masks, n_times=1):
        masks = np.tile(mask, (n_masks, 1))
        for i in range(n_times):
            masks = MaskOptimizer.perturb_masks(masks)
        return masks

    def perturb_masks(masks):
        def perturb_one_mask(mask):
            where_0 = np.nonzero(mask - 1)[0]
            where_1 = np.nonzero(mask)[0]
            i0 = np.random.randint(0, len(where_0), 1)
            i1 = np.random.randint(0, len(where_1), 1)
            mask[where_0[i0]] = 1
            mask[where_1[i1]] = 0
            return mask

        n_masks = len(masks)
        masks = np.apply_along_axis(perturb_one_mask, 1, masks)
        return masks

    def get_new_mask_batch(self, model, best_performing_mask,  gen_new_opt_mask):
        self.epoch_counter += 1
        random_masks = self.get_random_masks()

        if (gen_new_opt_mask):
            self.mask_opt = self.get_opt_mask(self.unmasked_data_size, model)

        if self.epoch_counter > 1:

            random_masks = MaskOptimizer.get_perturbed_masks(self.mask_opt, self.mask_batch_size, self.perturbation_size)
            index = int(self.frac_of_rand_masks * self.mask_batch_size)
            random_masks[index] = self.mask_opt
            random_masks[index + 1] = best_performing_mask

        return random_masks

    def get_mask_weights(self, tiling):
        w = np.ones(shape=self.mask_batch_size)
        index = int(self.frac_of_rand_masks * self.mask_batch_size)
        w[index] = 5
        w[index + 1] = 10
        return np.tile(w, tiling)
