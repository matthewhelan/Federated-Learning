# Federated-Learning

This is a Federated Learning run suite with FedAvg and Per-FedAvg (FO and HF versions) using torch.distributed. Special thanks to Xidong Wu of University of Pittsburgh for helping engineer the suite.

# Run

python main.py --dataset mnist0 --method perfedavg --worker-size 2 --epochs 10 --batch-size 5 --lr 0.005 --hessian 0 --init

on first run and 

python main.py --dataset mnist0 --method perfedavg --worker-size 2 --epochs 10 --batch-size 5 --lr 0.005 --hessian 0

for all runs after

Note: available methods are 'fedavg' and 'perfedavg', and Hessian approximation only for perfedavg, not fedavg

