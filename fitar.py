import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import os
import imageio

import argparse
import time
from colorama import init, Fore


init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--T", help="number of samples", type=int, default=1000)
parser.add_argument("--dmodel", help="d_model", type=int, default=32)
parser.add_argument("--nlayer", help="number of layers", type=int, default=3)
parser.add_argument("--gvideo", help="generate video", type=bool, default=False)
parser.add_argument("--window", help="window size", type=int, default=20)
parser.add_argument("--sigma", help="sigma in AR", type=float, default=1)
parser.add_argument("--signal", help="signal in AR", type=float, default=0)
parser.add_argument("--record_dir", help="directory to save record", type=str, default="")
parser.add_argument("--seed", help="number of layers", type=int, default=4869)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--epochs", help="number of epochs", type=int, default=1000)
parser.add_argument("--order", help="order of the AR model", type=int, default=2)
parser.add_argument("--dim", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.1)

args = parser.parse_args()

exp_name = f'sigma{args.sigma}_signal{args.signal}_seed{args.seed}_dim{args.dim}_order{args.order}'
exp_name += f'T{args.T}_dmodel{args.dmodel}_layer{args.nlayer}_window{args.window}_dropout{args.dropout}_lr{args.lr}_epochs{args.epochs}'

start_time = time.time()
def autoregressive_model(T, order, coeff=None, sigma=0.5, signal=0.3):
	if coeff is None:
		coeff = np.random.randn(order)
		coeff *= signal / np.sqrt(np.sum(np.square(coeff)))
	eps = sigma * np.random.randn(T)
	x = np.zeros(T + order)
	gt = np.zeros(T + order)

	for t in range(T):
		x[t + order] = eps[t] + np.sum(coeff * x[t : t+order])
		gt[t + order] = np.sum(coeff * x[t : t+order])
	return x[order: order + T], gt[order: order + T]


np.random.seed(args.seed)
T = args.T
order = args.order
x_array = []
oracle_loss = 0
for k in range(args.dim):
	x_k, _ = autoregressive_model(T, order, sigma=args.sigma, signal=args.signal)
	oracle_loss += np.mean(np.square(x_k - _)) / args.dim
	x_array.append(x_k)
x = np.transpose(np.array(x_array))
print(f'x: std = {np.std(x)}, range = ({np.min(x)}, {np.max(x)}), shape={np.shape(x)}')
print(f'x: null = {np.std(x)**2}, oracle={oracle_loss}')
split_time = T // 2
x_train = x[: split_time, :]
x_valid = x[split_time: , :]

window_size = args.window
batch_size = 30


def create_dataset(series, window_size):
	X, Y = [], []
	for i in range(len(series) - window_size):
		X.append(series[i:i+window_size, :])
		Y.append(series[i+window_size, :])
	X = np.array(X)
	Y = np.array(Y)
	print(f'create dataset: X shape = {np.shape(X)}, Y shape = {np.shape(Y)}')
	return torch.tensor(X).float(), torch.tensor(Y).float()

train_data, train_labels = create_dataset(x_train, window_size)
valid_data, valid_labels = create_dataset(x_valid, window_size)

train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Define Transformer model
class TransformerTimeSeries(nn.Module):
	def __init__(self, d_model, nhead, num_encoder_layers, dropout=0, input_dim=1, output_dim=1):
		super(TransformerTimeSeries, self).__init__()
		self.embedding = nn.Linear(input_dim, d_model)
		self.positionembedding = nn.Embedding(window_size, d_model)

		encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, batch_first=True, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
		
		self.fc = nn.Linear(d_model, output_dim)

	def forward(self, x):
		seq_len = x.size(1)
		batch_size = x.size(0)
		pos = torch.arange(0, seq_len, dtype=torch.long)
		x = self.embedding(x)
		x = x + self.positionembedding(pos).repeat(batch_size,1,1)
		x = self.transformer_encoder(x)
		x = self.fc(x[:, -1, :])
		return x

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class TrasnformerScheduler(_LRScheduler):
	def __init__(self, optimizer: Optimizer, dim_embed: int, warmup_steps: int,
					last_epoch: int=-1, verbose: bool=False) -> None:

		self.dim_embed = dim_embed
		self.warmup_steps = warmup_steps
		self.num_param_groups = len(optimizer.param_groups)

		super().__init__(optimizer, last_epoch, verbose)

	def get_lr(self) -> float:
		lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
		return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
	return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

# Model parameters
d_model = args.dmodel
nhead = 4
num_encoder_layers = args.nlayer
dropout = 0
epochs = args.epochs

model = TransformerTimeSeries(d_model, nhead, num_encoder_layers, dropout, args.dim, args.dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)#, betas = (0.9, 0.98), eps = 1.0e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs+1)
total_params = sum(p.numel() for p in model.parameters())

print(total_params)
image_paths = []

if args.gvideo:
	if not os.path.exists("predictions_images"):
		os.makedirs("predictions_images")

errors = np.ones((epochs, 2)) * 1e-5
for epoch in range(epochs):
	model.train()
	train_losses = []
	for batch_idx, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_losses.append(loss.item())
	
	with torch.no_grad():
		valid_output = model(valid_data)
		valid_loss = criterion(valid_output, valid_labels)

	errors[epoch, 0], errors[epoch, 1] = np.mean(train_losses), valid_loss.item()
	if errors[epoch, 0] < 1e-5:
		break
	#if epoch % 10 == 0:
	print(f'train loss = {errors[epoch, 0]}, valid loss = {errors[epoch, 1]}')

	if args.gvideo:
		# plot predictions
		model.eval()
		with torch.no_grad():
			train_predictions = model(train_data)
			valid_predictions = model(valid_data)

		plt.figure(figsize=(14, 7))

		# Plot actual series
		plt.plot(np.arange(split_time), x_train, label="Actual Training")
		plt.plot(np.arange(split_time, len(x)), x_valid, label="Actual Validation")
		# Plot predicted series
		plt.plot(np.arange(window_size, split_time), train_predictions.numpy(), label="Training Predictions")
		plt.plot(np.arange(split_time + window_size, len(x)), valid_predictions.numpy(), label="Validation Predictions")

		plt.title("Model Predictions on Training and Validation Sets: Epoch {}".format(epoch + 1))
		plt.xlabel("Time")
		plt.ylabel("Value")
		plt.legend()
		plt.grid(True)
		image_path = os.path.join("predictions_images", f"epoch_{epoch + 1}_predictions.png")
		plt.savefig(image_path)
		image_paths.append(image_path)
		plt.close()
	scheduler.step()

if len(args.record_dir) > 0:
	save_path = os.path.join(args.record_dir, exp_name + '.csv')
	np.savetxt(save_path, errors, delimiter=",")

if args.gvideo:
	video_path = "training_predictions_evolution.mp4"
	imageio.mimsave(video_path, [imageio.imread(image_path) for image_path in image_paths], fps=30)
	for image_path in image_paths:
		os.remove(image_path)

end_time = time.time()
print(f"Case {exp_name} done: time = {end_time - start_time} secs")