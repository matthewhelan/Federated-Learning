import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
import random
import torch.nn.functional as F
import numpy as np
import argparse
import time
import copy
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import itertools as it
from tqdm import tqdm
import pickle

loss_list = []
acc_list = []

#method to get pairs of data like "s -> (s0,s1), (s1,s2), (s2, s3), ..."
def pairwise(iterable):
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def PerFedAvg(train_set, test_set, model, args, device):

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    alpha_optimizer = torch.optim.SGD(model.parameters(), lr=args.alpha)
    beta_optimizer = torch.optim.SGD(model.parameters(), lr=args.beta)
    Iteration = 0
    rank = dist.get_rank()
    size = dist.get_world_size()

    t_toal = 0;
    t0 = time.time()
    counter = 0

    temp_model = copy.deepcopy(model)

    for epoch in range(args.epochs):
        #add progress bar with tqdm
        progress_bar = tqdm(train_set)
        for (data1, data2) in pairwise(progress_bar):
            counter += 1
            images1, labels1 = data1[0].to(device), data1[1].to(device)
            images2, labels2 = data2[0].to(device), data2[1].to(device)

            #copy the parameters
            temp_model = copy.deepcopy(model)

            if args.hessian: #Per-FedAvg(HF)

                model.train()

                #get the initial weights
                init_weights = list(map(lambda p: p, model.parameters()))

                #first step (Personalization step)
                output1 = model(images1)

                loss = F.nll_loss(output1, labels1)
                #delta = gradient(w_t)
                #instead of loss.backward(), calculate gradients with create_graph = True
                grad = torch.autograd.grad(loss, init_weights, create_graph=True)
                #SGD without .step() - ~w_t = w_t - alpha*delta
                updated_weights = list(map(lambda p: p[1] - args.alpha * p[0], zip(grad, init_weights)))

                # #second step (global update step)
                # #make parameters = updated_weights
                # # set model gradients 
                with torch.no_grad():
                    for p, g in zip(temp_model.parameters(), updated_weights):
                        p.copy_(g.detach().clone())

                temp_model.train()
                temp_model.zero_grad()

                output2 = temp_model(images2)

                loss2 = F.nll_loss(output2, labels2)

                #calculate gradients on updated weights
                grad_q = torch.autograd.grad(loss2, temp_model.parameters())
                #use grad_outputs
                # grad = torch.autograd.grad(updated_weights, model.parameters(), grad_outputs=grad_q)
                grad = torch.autograd.grad(updated_weights, init_weights, grad_outputs=grad_q)

                optimizer.zero_grad()
                # set model gradients 
                for p, g in zip(model.parameters(), grad):
                    p.grad = g.clone()
                #run SGD
                optimizer.step()

            else: #Per-FedAvg(FO)

                #first step (Personalization step)
                alpha_optimizer.zero_grad()

                #model.train()
                output1 = model(images1)

                loss = F.nll_loss(output1, labels1)
                #delta = gradient(w_t)
                loss.backward()


                beta_optimizer.zero_grad()
                
                # Update ~w_t = w_t - alpha*delta
                alpha_optimizer.step()

                #second step (global update step)
                model.train()

                output2 = model(images2)

                loss = F.nll_loss(output2, labels2)
                # compute ~delta = gradient of(~w_t)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(model.parameters(), temp_model.parameters()):
                    old_param.data = new_param.data.clone()

                # w_t+1 =  w_t - beta * ~delta
                beta_optimizer.step()

            if Iteration % args.inLoop == 0:
                #train_one_step(size, rank, model, test_set, args, device, Iteration)
                #### testing  #######
                model.eval()
                correct_cnt, ave_loss = 0, 0
                total_cnt = 0
                for batch_idx, (data, target) in enumerate(test_set):

                    data   = data.to(device)
                    target = target.to(device)

                    out = model(data)
                    _, pred_label = torch.max(out.data, 1)
                    total_cnt += data.data.size()[0]
                    correct_cnt += (pred_label == target.data).sum()

                #########
                loss_value = torch.tensor([loss.item(), correct_cnt, total_cnt], dtype=torch.float64)
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                loss_lst = loss_value.tolist()

                if rank == 0:
                    print("iteration", Iteration // args.inLoop, "training loss", loss_lst[0] / size, "test accuracy", loss_lst[1] / loss_lst[2] )  
                    loss_list.append(loss_lst[0] / size)
                    acc_list.append(loss_lst[1] / loss_lst[2])
                    if (Iteration // args.inLoop) % 200 == 0:
                        with open('train_loss_1_cifar.pkl', 'wb') as f:
                            pickle.dump(loss_list, f)
                        f.close()
                        with open('test_acc_1_cifar.pkl', 'wb') as f:
                            pickle.dump(acc_list, f)
                        f.close()
            Iteration += 1

            # ### Communication ############
            if Iteration % args.inLoop == 0:
                for para in model.parameters():
                    dist.all_reduce(para.data, op=dist.ReduceOp.SUM)
                    para.data.div_(size)
                
                
                        


#in this method, we run one step of local gradient descent, as the paper specifies
#then evaluate the model on the test set
def train_one_step(size, rank, model, test_set, args, device, Iteration):
    #train
    model.train()
    #copy the parameters
    temp_model = copy.deepcopy(list(model.parameters()))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.alpha)

    #iterate over the test set
    iter_loader = iter(test_set)
    # self.model.to(self.device)
    model.train()

    #get the next data
    (x, y) = next(iter_loader)

    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()

    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()

    optimizer.step()
    #evaluate
    model.eval()
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (data, target) in enumerate(test_set):

        data   = data.to(device)
        target = target.to(device)

        out = model(data)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += data.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()

    #########
    loss_value = torch.tensor([loss.item(), correct_cnt, total_cnt], dtype=torch.float64)
    dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
    loss_lst = loss_value.tolist()

    if rank == 0:
        print("Global model evaluation: itr", Iteration // args.inLoop, "training loss", loss_lst[0] / size, "test accuracy", loss_lst[1] / loss_lst[2] )
        loss_list.append(loss_lst[0] / size)
        acc_list.append(loss_lst[1] / loss_lst[2])
        if (Iteration // args.inLoop) % 200 == 0:
            with open('train_loss_1_cifar.pkl', 'wb') as f:
                pickle.dump(loss_list, f)
            f.close()
            with open('test_acc_1_cifar.pkl', 'wb') as f:
                pickle.dump(acc_list, f)
            f.close()



    # set local model back to client for training process
    for old_param, new_param in zip(model.parameters(), temp_model):
        old_param.data = new_param.data.clone()
