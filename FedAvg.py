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
import pickle

def FedAvg(train_set, test_set, model, args, device):

	optimizer = optim.SGD(model.parameters(), lr=args.lr)
	Iteration = 0
	rank = dist.get_rank()
	size = dist.get_world_size()

	t_toal = 0;
	t0 = time.time()

	loss_list = []
	acc_list = []	

	for epoch in range(args.epochs):
		for siter, (data, target) in enumerate(train_set):
			model.train()

			data   = data.to(device)
			target = target.to(device)

			output = model(data)

			loss = F.nll_loss(output, target)
			loss.backward()

            # Update
			optimizer.step()
			optimizer.zero_grad()

			if Iteration % args.inLoop == 0:

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
						with open('train_loss_fedavg_mn.pkl', 'wb') as f:
							pickle.dump(loss_list, f)
						f.close()
						with open('test_acc_fedavg_mn.pkl', 'wb') as f:
							pickle.dump(acc_list, f)
						f.close()

			Iteration += 1

			# ### Communication ############
			if Iteration % args.inLoop == 0:
				if rank == 0:
					t1 = time.time()

				for para in model.parameters():
					dist.all_reduce(para.data, op=dist.ReduceOp.SUM)
					para.data.div_(size)

