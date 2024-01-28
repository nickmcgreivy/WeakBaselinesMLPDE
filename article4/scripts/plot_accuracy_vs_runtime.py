import matplotlib.pyplot as plt
import numpy as onp
import h5py

from arguments import get_args

orders = [0, 1, 2]
nxs_dg = [[16, 32, 64, 128, 256], [8, 16, 32, 48, 64, 96, 128, 192], [8, 16, 24, 32, 48, 64, 96, 128]]
nxs_fv_baseline = [16, 32, 64, 128, 256, 512]
nxs_ps_baseline = [8, 16, 32, 64, 128, 256]

t_final = 10.0
outer_steps = int(t_final * 10)
t_chunk = t_final / outer_steps
N_test = 3 # average over 3 simulations
T_runtime = 1.0
device = 'cpu'

args = get_args()


### read runtimes
f1 = h5py.File(
	"{}/data/{}_fv.hdf5".format(args.read_write_dir, device),
	"r",
)
runtime_fv = []
for nx in nxs_fv_baseline:
	runtime_fv.append(f1[str(nx)][0] / T_runtime)
f1.close()

f1 = h5py.File(
	"{}/data/{}_ps.hdf5".format(args.read_write_dir, device),
	"r",
)
runtime_ps = []
for nx in nxs_ps_baseline:
	runtime_ps.append(f1[str(nx)][0] / T_runtime)
f1.close()

runtime_dg = []
for o, order in enumerate(orders):
	f2 = h5py.File("{}/data/{}_order{}.hdf5".format(args.read_write_dir, device, order), "r",)
	runtime_dg.append([])
	for nx in nxs_dg[o]:
		runtime_dg[o].append(f2[str(nx)][0] / T_runtime)
	f2.close()

### read correlation coefficients

fv_corr = onp.zeros((len(nxs_fv_baseline), outer_steps+1))
ps_corr = onp.zeros((len(nxs_fv_baseline), outer_steps+1))
order0_corr = onp.zeros((len(nxs_dg[0]), outer_steps+1))
order1_corr = onp.zeros((len(nxs_dg[1]), outer_steps+1))
order2_corr = onp.zeros((len(nxs_dg[2]), outer_steps+1))

for n in range(N_test):

	f_fv = h5py.File("{}/data/corr_run{}_fv.hdf5".format(args.read_write_dir, n),"r",)
	for i, nx in enumerate(nxs_fv_baseline):
		fv_corr[i] += f_fv[str(nx)][:] / N_test
	f_fv.close()

	f_ps = h5py.File("{}/data/corr_run{}_ps.hdf5".format(args.read_write_dir, n),"r",)
	for i, nx in enumerate(nxs_ps_baseline):
		ps_corr[i] += f_ps[str(nx)][:] / N_test
	f_ps.close()

	f0 = h5py.File("{}/data/corr_run{}_order0.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs_dg[0]):
		order0_corr[i] += f0[str(nx)][:] / N_test
	f0.close()

	f1 = h5py.File("{}/data/corr_run{}_order1.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs_dg[1]):
		order1_corr[i] += f1[str(nx)][:] / N_test
	f1.close()

	f2 = h5py.File("{}/data/corr_run{}_order2.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs_dg[2]):
		order2_corr[i] += f2[str(nx)][:] / N_test
	f2.close()


### find time until correlation < 0.95

t95_fv = [0.] * len(nxs_fv_baseline)
t95_ps = [0.] * len(nxs_ps_baseline)
t95_order0 = [0.] * len(nxs_dg[0])
t95_order1 = [0.] * len(nxs_dg[1])
t95_order2 = [0.] * len(nxs_dg[2])

for n in range(N_test):

	f_fv = h5py.File("{}/data/corr_run{}_fv.hdf5".format(args.read_write_dir, n),"r",)
	for i, nx in enumerate(nxs_fv_baseline):
		j = 0
		while True:
			if f_fv[str(nx)][j] < 0.95:
				break
			elif j >= (outer_steps):
				print("WARNING: NOT LESS THAN 0.95, FV, nx ={}".format(nx))
				break
			else:
				j += 1
		t95_fv[i] += ((j-1) * t_chunk + (0.95 - f_fv[str(nx)][j-1]) / (f_fv[str(nx)][j] - f_fv[str(nx)][j-1]) * t_chunk) / N_test
	f_fv.close()


	f_ps = h5py.File("{}/data/corr_run{}_ps.hdf5".format(args.read_write_dir, n),"r",)
	for i, nx in enumerate(nxs_ps_baseline):
		j = 0
		while True:
			if f_ps[str(nx)][j] < 0.95:
				break
			elif j >= (outer_steps):
				print("WARNING: NOT LESS THAN 0.95, PS, nx ={}".format(nx))
				break
			else:
				j += 1
		t95_ps[i] += ((j-1) * t_chunk + (0.95 - f_ps[str(nx)][j-1]) / (f_ps[str(nx)][j] - f_ps[str(nx)][j-1]) * t_chunk) / N_test
	f_ps.close()


	f0 = h5py.File("{}/data/corr_run{}_order0.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs_dg[0]):
		j = 0
		while True:
			if f0[str(nx)][j] < 0.95:
				break
			elif j >= (outer_steps):
				print("WARNING: NOT LESS THAN 0.95, order={}, nx ={}".format(1, nx))
				break
			else:
				j += 1
		t95_order0[i] += ((j-1) * t_chunk + (0.95 - f0[str(nx)][j-1]) / (f0[str(nx)][j] - f0[str(nx)][j-1]) * t_chunk) / N_test
	f0.close()

	f1 = h5py.File("{}/data/corr_run{}_order1.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs_dg[1]):
		j = 0
		while True:
			if f1[str(nx)][j] < 0.95:
				break
			elif j >= (outer_steps):
				print("WARNING: NOT LESS THAN 0.95, order={}, nx ={}".format(1, nx))
				break
			else:
				j += 1
		t95_order1[i] += ((j-1) * t_chunk + (0.95 - f1[str(nx)][j-1]) / (f1[str(nx)][j] - f1[str(nx)][j-1]) * t_chunk) / N_test
	f1.close()

	f2 = h5py.File("{}/data/corr_run{}_order2.hdf5".format(args.read_write_dir, n),"r")
	for i, nx in enumerate(nxs_dg[2]):
		j = 0
		while True:
			if f2[str(nx)][j] < 0.95:
				break
			elif j >= (outer_steps):
				print("WARNING: NOT LESS THAN 0.95, order={}, nx ={}".format(2, nx))
				break
			else:
				j += 1
		t95_order2[i] += ((j-1) * t_chunk + (0.95 - f2[str(nx)][j-1]) / (f2[str(nx)][j] - f2[str(nx)][j-1]) * t_chunk) / N_test
	f2.close()

#### Plot 1: time vs correlation
fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots()
fig3, axs3 = plt.subplots()

T = onp.arange(0, outer_steps+1) * t_chunk


#for i, nx in enumerate(nxs_fv_baseline):
#	axs1.plot(T, fv_corr[i], label="FV nx={}".format(nx))











color_ps = ['#03ffb5', '#34cc7e', '#389c4f', '#2e6e25','#1e4400']
color_dg = ['#ff7a03', '#d1520c', '#a12b0a', '#710101']
lw = 3.0


nxs_dg_plot1 = [0, 1, 3, 5]
for i, nx in enumerate(nxs_ps_baseline[1:]):
	axs1.plot(T, ps_corr[i+1], label="PS nx={}".format(nx), color=color_ps[i], linewidth=lw)

for i, j in enumerate(nxs_dg_plot1):
	axs1.plot(T, order2_corr[j], label="DG nx={}".format(nxs_dg[2][j]), color=color_dg[i], linewidth=lw, linestyle='-.')

axs1.spines['top'].set_visible(False)
axs1.spines['right'].set_visible(False)
axs1.set_xlabel("Simulation time", fontsize=14)
axs1.set_ylabel("Vorticity correlation", fontsize=14)
axs1.set_xlim([0,10.0])
axs1.set_ylim([0,1.01])
fig1.legend(loc=(0.15, 0.15), frameon=False)

#### Plot 2: grid resolution vs t95

axs2.plot(nxs_fv_baseline, t95_fv, label="FV")
axs2.plot(nxs_ps_baseline, t95_ps, label="PS")
axs2.plot(nxs_dg[0], t95_order0, label="DG p=0")
axs2.plot(nxs_dg[1], t95_order1, label="DG p=1")
axs2.plot(nxs_dg[2], t95_order2, label="DG p=2")
axs2.set_xlabel("grid resolution")
axs2.set_xscale('log')
axs2.set_ylabel("T95")
axs2.set_ylim([0.,10.0])
axs2.set_xticks([])
axs2.set_xticks([8, 16, 32, 64, 128, 256, 512])
axs2.minorticks_off()
axs2.set_xticklabels(['8', '16', '32', '64', '128', '256', '512'])
fig2.legend()



#### Plot 3: t95 vs runtime

ms = 15.0
axs3.plot(t95_fv[2:], runtime_fv[2:], label="Finite Volume")
axs3.scatter(t95_fv[2:], runtime_fv[2:], s=ms, label="Finite Volume")
axs3.plot(t95_ps[1:-1], runtime_ps[1:-1], label="Pseudospectral")
axs3.scatter(t95_ps[1:-1], runtime_ps[1:-1], s=ms, label="Pseudospectral")
#axs3.plot(t95_order0, runtime_dg[0], label="DG p=0")
#axs3.plot(t95_order1, runtime_dg[1], label="DG p=1")
axs3.plot(t95_order2[:-3], runtime_dg[2][:-3], label="Discontinuous Galerkin")
axs3.scatter(t95_order2[:-3], runtime_dg[2][:-3], s=ms, label="Discontinuous Galerkin")

axs3.spines['top'].set_visible(False)
axs3.spines['right'].set_visible(False)
axs3.set_xlabel("Time until correlation $<$ 0.95")
axs3.set_ylabel("Runtime per unit time (s)")
axs3.set_yscale('log')
axs3.set_xlim([0.,10.])
axs3.set_ylim([1e-2-6*1e-3,10+4])

axs3.text(0.52, 1.03, 'Finite Volume', transform=axs3.transAxes, fontsize=14,
        verticalalignment='top')
axs3.text(0.08, 0.22, r'$64$' "\n" r'$\times64$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.26, 0.57, r'$128$' "\n" r'$\times128$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.37, 0.78, r'$256$' "\n" r'$\times256$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.42, 1.0, r'$512$' "\n" r'$\times512$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.78, 0.74, 'Pseudospectral', transform=axs3.transAxes, fontsize=14,
        verticalalignment='top')
axs3.text(0.04, 0.1, r'$16$' "\n" r'$\times16$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.03, 0.33, r'$32$' "\n" r'$\times32$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.37, 0.53, r'$64$' "\n" r'$\times64$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.8, 0.65, r'$128$' "\n" r'$\times128$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.63, 0.87, 'Discontinuous Galerkin', transform=axs3.transAxes, fontsize=14,
        verticalalignment='top')
axs3.text(0.26, 0.1, r'$8$' "\n" r'$\times8$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.39, 0.34, r'$16$' "\n" r'$\times16$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.47, 0.61, r'$24$' "\n" r'$\times24$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.56, 0.7, r'$32$' "\n" r'$\times32$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')
axs3.text(0.68, 0.82, r'$48$' "\n" r'$\times48$', transform=axs3.transAxes, fontsize=10,
        verticalalignment='top')


plt.show()
