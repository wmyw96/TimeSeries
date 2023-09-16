import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

dim_cand = [16, 32, 64, 128]
T_cand = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#9acdc4',  # pain blue
	'#05348b',  # dark blue
	'#6bb392',
	'#e5a84b',
]

marker = [
	'8',
	's',
	'x',
	'^',
]

'''
results = []
for (i, dim) in enumerate(dim_cand):
	results.append([])
	for (j, T) in enumerate(T_cand):
		exp_name = f'T{T}_dim{dim}_layer{3}_window{20}_sigma{1}_signal{0}_seed{0}_lr{0.0005}'
		content = genfromtxt(f"logs/{exp_name}.csv", delimiter=',')
		print(f'i={i}, j={j}, shape={np.shape(content)}')
		results[i].append(content)


cmap = plt.get_cmap("RdBu_r")
plt.figure(figsize=(16, 6))
for i in [2]:
	for (j, T) in enumerate(T_cand):
		epoches = np.shape(results[i][j])[0]
		cand_epoch = np.arange(epoches)
		plt.plot(cand_epoch, results[i][j][:, 0], color=cmap(j/(len(T_cand)-1)), linestyle=lines[i])
'''

for (i, dim) in enumerate(dim_cand):
	min_l2_loss = np.zeros(len(T_cand))
	for (j, T) in enumerate(T_cand):
		l2_loss = []
		for lr in [0.0001, 0.0002, 0.0005]:
			exp_name = f'T{T}_dim{dim}_layer{3}_window{20}_sigma{1}_signal{0}_seed{0}_lr{lr}'
			try:
				content = genfromtxt(f"logs/{exp_name}.csv", delimiter=',')
				l2_loss.append(np.maximum(np.min(content[:, 0]), 1e-5))
			except:
				print(f'file not find{exp_name}')
		min_l2_loss[j] = np.min(l2_loss)
	plt.plot(T_cand, min_l2_loss, color=color_tuple[i], marker=marker[i], label=f'dmodel={dim}')


plt.ylabel(r"loss", fontsize=22)
plt.xlabel(r"T", fontsize=22)
plt.yscale("log")
plt.legend(loc='best')
#plt.xscale("log")
plt.show()