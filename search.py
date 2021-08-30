# general use
import torch
import torch.optim as op
import numpy as np


from sklearn.model_selection import train_test_split
from data import load_data_with_features_continious
from models.searchable_model import Searchable_DKT_COMB

# surrogate related
import models.surrogate as surr
from train_search import  train_sampled_models
from train_search import update_surrogate_dataloader, train_surrogate, predict_accuracies_with_surrogate

import utils

# %%
"""
 Base class for NAS
"""


class ModelSearcher():
    def __init__(self, args):
        self.args = args

    def search(self):
        pass

    def _epnas(self, model_type, surrogate_dict, dataloaders, device):

        # surrogate
        surrogate = surrogate_dict['model']
        s_crite = surrogate_dict['criterion']
        s_data = surr.SurrogateDataloader()
        s_optim = op.Adam(surrogate.parameters(), lr=self.args.lr_surrogate)


        temperature = self.args.initial_temperature
        sampled_k_confs = []
        shared_weights = dict()

        # repeat process search_iterations times
        for si in range(self.args.search_iterations):

            if self.args.verbose:
                print(50 * "=")
                print("Search iteration {}/{} ".format(si, self.args.search_iterations))

            # for each fusion
            for progression_index in range(self.args.max_progression_levels):

                if self.args.verbose:
                    print(25 * "-")
                    print("Progressive step {}/{} ".format(progression_index, self.args.max_progression_levels))

                    # Step 1: unfold layer (fusion index)
                list_possible_layer_confs = utils.get_possible_layer_configurations(progression_index)

                # Step 2: merge previous top with unfolded configurations
                all_configurations = utils.merge_unfolded_with_sampled(sampled_k_confs, list_possible_layer_confs,
                                                                       progression_index)

                # Step 3: obtain accuracies for all possible unfolded configurations
                # if first execution, just train all, if not, use surrogate to predict them
                if si + progression_index == 0:
                    all_accuracies = train_sampled_models(all_configurations, model_type, dataloaders, self.args,
                                                          device, state_dict=shared_weights)
                    update_surrogate_dataloader(s_data, all_configurations, all_accuracies)
                    train_surrogate(surrogate, s_data, s_optim, s_crite, self.args, device)

                    if self.args.verbose:
                        print("Trained architectures: ")
                        print(list(zip(all_configurations, all_accuracies)))
                else:
                    all_accuracies = predict_accuracies_with_surrogate(all_configurations, surrogate, device)
                    if self.args.verbose:
                        print("Predicted accuracies: ")
                        print(list(zip(all_configurations, all_accuracies)))

                # Step 4: sample K architectures and train them.
                # this should happen only if not first iteration because in that case,
                # all confs were trained in step 3
                if si + progression_index == 0:
                    sampled_k_confs = utils.sample_k_configurations(all_configurations, all_accuracies,
                                                                    self.args.num_samples, temperature)

                    if self.args.verbose:
                        estimated_accuracies = predict_accuracies_with_surrogate(all_configurations, surrogate,
                                                                                       device)
                        diff = np.abs(np.array(estimated_accuracies) - np.array(all_accuracies))
                        print("Error on accuracies = {}".format(diff))

                else:
                    sampled_k_confs = utils.sample_k_configurations(all_configurations, all_accuracies,
                                                                    self.args.num_samples, temperature)
                    sampled_k_accs = train_sampled_models(sampled_k_confs, model_type, dataloaders, self.args, device,
                                                          state_dict=shared_weights)

                    update_surrogate_dataloader(s_data, sampled_k_confs, sampled_k_accs)
                    err = train_surrogate(surrogate, s_data, s_optim, s_crite, self.args, device)

                    if self.args.verbose:
                        print("Trained architectures: ")
                        print(list(zip(sampled_k_confs, sampled_k_accs)))
                        print("with surrogate error: {}".format(err))

                # temperature decays at each step
                iteration = si * self.args.search_iterations + progression_index
                temperature = utils.compute_temperature(iteration, self.args)
                if self.args.verbose:
                    print("Temperature is being set to {}".format(temperature))

        return s_data

    def _randsearch(self, model_type, dataloaders, dataset_searchmethods, device):

        # surrogate (in here, we only use the dataloader as means to keep track of real accuracies during exploration)
        s_data = surr.SurrogateDataloader()

        # search functions that are specific to the dataset
        train_sampled_models = dataset_searchmethods['train_sampled_fun']
        get_possible_layer_configurations = dataset_searchmethods['get_layer_confs']

        sampled_k_confs = []

        shared_weights = dict()

        # repeat process search_iterations times
        for si in range(self.args.search_iterations * self.args.max_progression_levels):

            if self.args.verbose:
                print(50 * "=")
                print("Random Search iteration {}/{} ".format(si,
                                                              self.args.search_iterations * self.args.max_progression_levels))

            # Step 1: sample
            sampled_k_confs = utils.sample_k_configurations_directly(self.args.num_samples,
                                                                     self.args.max_progression_levels,
                                                                     get_possible_layer_configurations)
            sampled_k_accs = train_sampled_models(sampled_k_confs, model_type, dataloaders, self.args, device,
                                                  state_dict=shared_weights)

            # Step 2: keep accuracy measure
            update_surrogate_dataloader(s_data, sampled_k_confs, sampled_k_accs)

            if self.args.verbose:
                print("Trained architectures: ")
                print(list(zip(sampled_k_confs, sampled_k_accs)))

        return s_data


class KTSearcher(ModelSearcher):
    def __init__(self, args, device):
        super(KTSearcher, self).__init__(args)

        self.device = device

        self.dataset, self.all_skills, self.num_steps = load_data_with_features_continious(args.data_path)
        self.num_skills = len(self.all_skills)
        self.train_data, self.valid_data = train_test_split(self.dataset, test_size=0.5, random_state=3)
        #datasets = {'train': self.train_data, 'dev': self.valid_data}

    def search(self):
        surrogate = surr.SimpleRecurrentSurrogate(100, 2, 100)
        surrogate.to(self.device)
        surrogate_dict = {'model': surrogate, 'criterion': torch.nn.MSELoss()}
        datasets_dict = {'train': self.train_data, 'val': self.valid_data, 'all_skills': self.all_skills, 'num_steps':
                         self.num_steps, 'num_skills': self.num_skills}

        return self._epnas(Searchable_DKT_COMB, surrogate_dict, datasets_dict,
                           self.device)
