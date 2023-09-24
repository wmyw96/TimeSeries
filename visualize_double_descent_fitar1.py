import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

dim_cand = [5, 10, 20]
dmodel_cand = [16, 24, 32, 40, 48, 56, 64, 72, 96, 128]#, 160, 192, 224, 256]#, 14000, 16000, 18000, 20000]
num_params = [7338, 15610, 26954, 41370, 58858, 79418, 103050, 129754, 228298, 402698]#, 626250, 898954, 1220810, 1591818]

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
	for (j, dmodel) in enumerate(dmodel_cand):
		exp_name = f'sigma{0.3}_signal{0.95}_seed{0}_dim{dim}_order{1}_'
		exp_name += f'T{5000}_dmodel{dmodel}_layer{3}_window{20}_dropout{0.0}_lr{0.001}_epochs{1000}'
		try:
			content = genfromtxt(f"logs/{exp_name}.csv", delimiter=',')
			l2_loss_train.append(np.maximum(np.min(content[:, 0]), 1e-5))
			l2_loss_test.append(content[np.argmin(content[:, 1]), 2])
			#l2_loss_test.append(np.maximum(np.min(content[:, 2]), 1e-5))
			l2_loss_test2.append(content[-1, 1])
		except:
			print(f'file not find{exp_name}')

	ax1.plot(np.array(num_params, dtype=np.float) / (5000*dim), l2_loss_train, color=color_tuple[i], marker=marker[i], linestyle='dotted', label=f'dim={dim}')
	ax2.plot(np.array(num_params, dtype=np.float) / (5000*dim), l2_loss_test, color=color_tuple[i], marker=marker[i], label=f'dim={dim}')
	ax2.plot(np.array(num_params, dtype=np.float) / (5000*dim), l2_loss_test2, color=color_tuple[i], marker=marker[i], linestyle='dotted', label=f'dim={dim}')




ax1.set_ylabel(r"train loss", fontsize=22)
ax1.set_xlabel(r"# params / # of constraints", fontsize=22)
ax1.legend(loc='best')
ax1.set_yscale("log")
ax1.axvline(x=1, color='black', linestyle='dotted')
ax1.set_xscale("log")

ax2.set_ylabel(r"test loss", fontsize=22)
ax2.set_xlabel(r"# params / # of constraints", fontsize=22)
ax2.axhline(y=0.09, color='black', linestyle='-')
ax2.axvline(x=1, color='black', linestyle='dotted')
ax2.legend(loc='best')
ax2.set_yscale("log")
ax2.set_yticks([0.09, 0.10, 0.15, 0.30, 0.50, 0.92], [0.09, 0.10, 0.15, 0.30, 0.50, 0.92])
ax2.set_xscale("log")

plt.show()