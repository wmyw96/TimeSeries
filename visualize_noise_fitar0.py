import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

dim_cand = [5, 10, 20]
T_cand = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000]#, 14000, 16000, 18000, 20000]

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

plt.figure(figsize=(16, 8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for (i, dim) in enumerate(dim_cand):
	l2_loss_train = []
	l2_loss_test = []	
	l2_loss_test2 = []
	for (j, T) in enumerate(T_cand):
		exp_name = f'sigma{0.3}_signal{0.95}_seed{0}_dim{dim}_order{1}'
		exp_name += f'T{T}_dmodel{32}_layer{3}_window{20}_dropout{0.1}_lr{0.001}_epochs{1000}'
		try:
			content = genfromtxt(f"logs/{exp_name}.csv", delimiter=',')
			l2_loss_train.append(np.maximum(np.min(content[:, 0]), 1e-5))
			l2_loss_test.append(np.maximum(np.min(content[:, 1]), 1e-5))
			l2_loss_test2.append(content[-1, 1])
		except:
			print(f'file not find{exp_name}')

	ax1.plot(np.array(T_cand) * dim, l2_loss_train, color=color_tuple[i], marker=marker[i], linestyle='dotted', label=f'dim={dim}')
	ax2.plot(np.array(T_cand) * dim, l2_loss_test, color=color_tuple[i], marker=marker[i], label=f'dim={dim}')
	ax2.plot(np.array(T_cand) * dim, l2_loss_test2, color=color_tuple[i], marker=marker[i], linestyle='dotted', label=f'dim={dim}')




ax1.set_ylabel(r"train loss", fontsize=22)
ax1.set_xlabel(r"# of constraints", fontsize=22)
ax1.legend(loc='best')
ax1.set_yscale("log")
ax1.axvline(x=26900, color='black', linestyle='dotted')
ax1.set_xscale("log")

ax2.set_ylabel(r"test loss", fontsize=22)
ax2.set_xlabel(r"# of constraints", fontsize=22)
ax2.axhline(y=0.09, color='black', linestyle='-')
ax2.axvline(x=26900, color='black', linestyle='dotted')
ax2.legend(loc='best')
ax2.set_yscale("log")
ax2.set_yticks([0.09, 0.10, 0.15, 0.30, 0.50, 0.92], [0.09, 0.10, 0.15, 0.30, 0.50, 0.92])
ax2.set_xscale("log")

plt.show()