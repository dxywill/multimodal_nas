import torch
import torch.nn as nn
import torch.optim as op
import os

import numpy as np
import copy
from typing import List

from sklearn import metrics
from scipy.stats import pearsonr

import models.surrogate as surr

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train_sampled_models(sampled_configurations, searchable_type, dataloaders,
                         args, device,
                         return_model=[], premodels=[], preaccuracies=[],
                         train_only_central_params=True,
                         state_dict=dict()):
    use_weightsharing = args.weightsharing

    num_steps = dataloaders['num_steps']
    num_skills = dataloaders['num_skills']
    criterion = torch.nn.CrossEntropyLoss()
    real_accuracies = []

    if return_model:
        models = []

    for idx, configuration in enumerate(sampled_configurations):

        if not return_model or idx in return_model:

            # model to train
            # if not premodels:
            #     dkt_model = searchable_type(args, configuration)
            #     # loading pretrained weights
            #     dkt_model_filename = os.path.join(args.checkpointdir, args.ske_cp)
            #
            #     dkt_model.load_state_dict(torch.load(dkt_model_filename))
            # else:
            dkt_model = searchable_type(num_skills, num_steps, configuration, device)
            # Load weights from a pre-trained model
            #dkt_model.load_state_dict(torch.load('./model_dkt_gpu.pth'))

                # if args.use_dataparallel:
                #     rmode.load_state_dict(premodels[idx].module.state_dict())
                # else:
                #     rmode.load_state_dict(premodels[idx].state_dict())

                    # parameters to update during training


            # optimizer and scheduler
            optimizer = op.Adam(dkt_model.parameters(), lr=args.eta_max, weight_decay=1e-4)
            #scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
            #                                          num_batches_per_epoch)

            # hardware tuning
            # if torch.cuda.device_count() > 1 and args.use_dataparallel:
            #     rmode = torch.nn.DataParallel(rmode)
            dkt_model.to(device)
            #
            # if use_weightsharing:
            #     set_central_states(rmode, state_dict, args.use_dataparallel)

            if args.verbose:
                print('Now training: ')
                print(configuration)

            if not preaccuracies:
                best_model_acc = train_dkt_track_acc(dkt_model, criterion, optimizer, dataloaders, args.batchsize,
                                                        device=device, num_epochs=args.epochs, verbose=args.verbose)
            else:
                best_model_acc = train_dkt_track_acc(dkt_model, criterion, optimizer,  dataloaders, args.batchsize,
                                                        device=device, num_epochs=args.epochs, verbose=args.verbose,
                                                        init_f1=preaccuracies[idx])

            # if use_weightsharing:
            #     state_dict = get_central_states(rmode, state_dict, args.use_dataparallel)

            real_accuracies.append(best_model_acc)

            if return_model:
                models.append(dkt_model)

    if return_model:
        return real_accuracies, models
    else:
        return real_accuracies



# %% for simple multimodal
def train_dkt_track_acc(model, criteria, optimizer, dataloaders, batch_size=32,
                        device=None, num_epochs=200, verbose=False, multitask=False):
    best_model_sd = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    max_grad_norm = 20

    train_data = dataloaders['train']
    val_data = dataloaders['val']
    all_skills = dataloaders['all_skills']
    num_steps = dataloaders['num_steps']
    num_skills= dataloaders['num_skills']

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:

            if phase == 'train':
                data = train_data
            else:
                data = val_data

            running_loss = 0.0
            running_auc = 0
            running_r2 = 0

            # Iterate over data.
            # count = 0
            num_batches = len(data) // batch_size
            idx = 0
            actual_labels = []
            pred_labels = []
            hidden = model.init_hidden(batch_size).to(device)
            #hidden = (h.to(device) for h in hidden)
            for b in range(1):
            #for b in range(num_batches - 1):

                inputs, current_x,  targets_id, targets_correctness, num_problems = get_batch(data, idx, batch_size,
                                                                num_steps, num_skills, all_skills)

                idx += batch_size
                # device
                actual_labels += targets_correctness.tolist()
                inputs = (inp.to(device) for inp in inputs)
                current_x = current_x.to(device)
                targets_id = targets_id.to(device)
                targets_correctness = targets_correctness.to(device)

                if phase == 'train':
                    hidden = repackage_hidden(hidden)
                    # zero the parameter gradients
                    model.train()
                    optimizer.zero_grad()

                    output, hidden = model(inputs, hidden, routers_info=current_x)


                    output = output.contiguous().view(-1)
                    logits = torch.gather(output, 0, targets_id)

                    # logits = slip_guess(logits)
                    # preds
                    preds = torch.sigmoid(logits)
                    for p in preds:
                        pred_labels.append(p.item())

                    # criterion = nn.CrossEntropyLoss()
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(logits, targets_correctness)
                    # loss = weighted_loss(num_problems, logits, target_correctness, criterion)
                    # loss = time_weighted_loss(num_problems, logits, target_correctness)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                    running_loss += loss.item()
                else:
                    with torch.no_grad():
                        hidden = repackage_hidden(hidden)
                        # zero the parameter gradients
                        model.eval()

                        output, hidden = model(inputs, hidden, routers_info=current_x)

                        output = output.contiguous().view(-1)
                        logits = torch.gather(output, 0, targets_id)

                        # logits = slip_guess(logits)
                        # preds
                        preds = torch.sigmoid(logits)
                        for p in preds:
                            pred_labels.append(p.item())
                        # criterion = nn.CrossEntropyLoss()
                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(logits, targets_correctness)
                        running_loss += loss.item()

                # print pred_labels
                #rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
                fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                # calculate r^2
                #r2 = r2_score(actual_labels, pred_labels)
                # pearson r
                r_row, p_value = pearsonr(actual_labels, pred_labels)

                #print("Epoch: {},  Batch {}/{} AUC: {} r^2: {}".format(epoch, count, batch_num, auc, r_row * r_row))

                running_auc = auc
                running_r2  = r_row * r_row

                #running_corrects += torch.sum(preds == label.data)

            epoch_loss = running_loss / num_batches
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} AUC: {:.4f} R2: {:.4f}'.format(
                phase, epoch_loss, running_auc, running_r2))

            # deep copy the model
            if phase == 'dev' and running_auc > best_auc:
                best_auc = running_auc
                best_model_sd = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_sd)
    model.train(False)

    return best_auc


# %% for simple multimodal
def test_dkt_track_acc(model, dataloaders, dataset_sizes,
                       device=None, multitask=False):
    model.train(False)
    phase = 'test'

    running_corrects = 0

    # Iterate over data.
    for data in dataloaders[phase]:

        # get the inputs
        rgb, ske, label = data['rgb'], data['ske'], data['label']

        # device
        rgb = rgb.to(device)
        ske = ske.to(device)
        label = label.to(device)

        output = model((rgb, ske))

        if not multitask:
            _, preds = torch.max(output, 1)
        else:
            _, preds = torch.max(sum(output), 1)

            # statistics
        running_corrects += torch.sum(preds == label.data)

    acc = running_corrects.double() / dataset_sizes[phase]

    return acc


def update_surrogate_dataloader(surrogate_dataloader, configurations, accuracies):
    for conf, acc in zip(configurations, accuracies):
        with open('results.csv', 'a') as the_file:
            the_file.write(str(conf) + ', ' + str(acc) + '\n')
        surrogate_dataloader.add_datum(conf, acc)


def train_surrogate(surrogate, surrogate_dataloader, surrogate_optimizer, surrogate_criterion, args, device):
    s_data = surrogate_dataloader.get_data(to_torch=True)
    err = surr.train_simple_surrogate(surrogate, surrogate_criterion,
                                      surrogate_optimizer, s_data,
                                      args.epochs_surrogate, device)

    return err


def predict_accuracies_with_surrogate(configurations, surrogate, device):
    # uses surrogate to evaluate input configurations

    accs = []

    for c in configurations:
        accs.append(surrogate.eval_model(c, device))

    return accs

def get_batch(source, idx, batch_size, num_steps, num_skills, all_skills):

    input_size = num_skills * 2
    x_1 = np.zeros((batch_size, num_steps))
    x_2 = np.zeros((batch_size, num_steps))
    x_3 = np.zeros((batch_size, num_steps))
    x_4 = np.zeros((batch_size, num_steps))
    x_current = np.zeros((batch_size, num_steps))

    target_id: List[int] = []
    target_correctness = []
    num_problems = []
    for i in range(batch_size):
        student_tries = source[idx + i]
        num_problems.append(len(student_tries) - 1)

        for j in range(len(student_tries) - 1):
            problem_index = int(student_tries[j][0])
            problem_id = all_skills.index(problem_index)
            label_index = 0
            if (int(student_tries[j][1]) == 0):
                label_index = 0
            else:
                label_index = 1

            x_1[i, j] = label_index
            x_2[i, j] = float(student_tries[j][3])
            x_3[i, j] = float(student_tries[j][4])
            x_4[i, j] = float(student_tries[j][5])
            x_current[i, j] = problem_id
            next_problem_index = student_tries[j + 1][0]
            next_problem_id = all_skills.index(next_problem_index)
            target_id.append(i * num_steps * num_skills + j * num_skills + int(next_problem_id))
            target_correctness.append(int(student_tries[j + 1][1]))

    target_id = torch.tensor(target_id, dtype=torch.int64)
    target_correctness = torch.tensor(target_correctness, dtype=torch.float)

    x_1 = torch.tensor(x_1, dtype=torch.int64)
    x_1 = torch.unsqueeze(x_1, 2)
    input_data = torch.FloatTensor(batch_size, num_steps, 2)
    input_data.zero_()
    input_data.scatter_(2, x_1, 1)

    x_2 = torch.tensor(x_2, dtype=torch.float)
    x_2 = x_2.unsqueeze(2)

    x_3 = torch.tensor(x_3, dtype=torch.float)
    x_3 = x_3.unsqueeze(2)

    x_4 = torch.tensor(x_4, dtype=torch.float)
    x_4 = x_4.unsqueeze(2)

    x_current = torch.tensor(x_current, dtype=torch.int64)
    input_data = (input_data, x_2, x_3, x_4)

    return input_data, x_current, target_id, target_correctness, num_problems