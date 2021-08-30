# general use
import torch
import numpy as np
import random

"""
 Auxiliary functions for correct exploration of Searchable Models 
"""


def get_possible_layer_configurations(progression_index):

    list_conf = []
    #max_labels = [3 + progression_index, 4 + progression_index, 3]

    for ti in range(4 + progression_index):
        for vi in range(0, 3):
            conf = [ti, vi]
            list_conf.append(conf)

    return list_conf


def sample_k_configurations(configurations, accuracies_, k, temperature):
    accuracies = np.array(accuracies_)
    p = accuracies / accuracies.sum()
    powered = pow(p, 1.0 / temperature)
    p = powered / powered.sum()

    if len(configurations) > k:
        try:
            indices = np.random.choice(len(configurations), k, replace=False, p=p)
        except:
            indices = np.random.choice(len(configurations), k, replace=False)

        samples = [configurations[i] for i in indices]
    else:
        samples = configurations

    return samples


def sample_k_configurations_uniform(configurations, k):
    indices = np.random.choice(len(configurations), k)
    samples = [configurations[i] for i in indices]

    return samples


def merge_unfolded_with_sampled(previous_top_k_configurations, unfolded_configurations, layer):
    # normally, the outpout configurations are evaluated with the surrogate function

    # unfolded_configurations is a configuration for a single layer, so it does not have seq_len dimension
    # previous_top_k_configurations is composed of configurations of size (seq_len,3)
    merged = []

    if not previous_top_k_configurations:
        # this typically executes at the very first iteration of the sequential exploration
        for unfolded_conf in unfolded_configurations:

            if layer == 0:
                new_conf = np.expand_dims(unfolded_conf, 0)
            else:
                raise ValueError(
                    'merge_unfolded_with_sampled: Something weird is happening. previous_top_k_configurations is None, but layer != 0')

            merged.append(new_conf)
    else:
        # most common pathway of execution: there exist previous configurations
        for prev_conf in previous_top_k_configurations:
            for unfolded_conf in unfolded_configurations:
                # nodes = []
                # for s in prev_conf:
                #     nodes.append(s[0])
                # if  unfolded_conf[0] in nodes:
                #     continue
                new_conf = np.concatenate([prev_conf, np.expand_dims(unfolded_conf, 0)], 0)
                merged.append(new_conf)
    # filtered = []
    # for m in merged:
    #     nodes = []
    #     for n in m:
    #         nodes.append(n[0])
    #     if 0 in nodes:
    #         filtered.append(m)

    return merged


def sample_k_configurations_directly(k, max_progression_levels, get_possible_layer_configurations_fun):
    configurations = []

    possible_confs_per_layer = []
    for l in range(max_progression_levels):
        possible_confs_per_layer.append(get_possible_layer_configurations_fun(l))

    for sample in range(k):
        num_layers_sample = random.randint(1, max_progression_levels)

        conf = []
        for layer in range(num_layers_sample):
            random_layer_conf = sample_k_configurations_uniform(possible_confs_per_layer[l], 1)
            conf.append(random_layer_conf)

        conf = np.array(conf)[:, 0, :]
        configurations.append(conf)

    return configurations


def compute_temperature(iteration, args):
    temp = (args.initial_temperature - args.final_temperature) * np.exp(
        -(iteration + 1.0) ** 2 / args.temperature_decay ** 2) + args.final_temperature
    return temp


# use the next 3 functions to initial a model, eg at the third function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

        # Initialize biases for LSTM’s forget gate to 1 to remember more by default. Similarly, initialize biases for GRU’s reset gate to -1.
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    elif classname.find('GRU') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)


def initial_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            weights_init(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight([sub_layer])