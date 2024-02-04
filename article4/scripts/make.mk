# SCRIPTS
COMPUTE_RUNTIME_SCRIPT = $(BASEDIR)/scripts/compute_runtime.py
COMPUTE_RUNTIME_GPUTPU_SCRIPT = $(BASEDIR)/scripts/compute_runtime_gputpu.py
COMPUTE_CORRCOEF_SCRIPT = $(BASEDIR)/scripts/compute_corrcoef.py
DG_PLOT_SIM = $(BASEDIR)/scripts/plot_dg_demo.py
PLOT_ACCURACY_RUNTIME_SCRIPT = $(BASEDIR)/scripts/plot_accuracy_vs_runtime.py

ARGS = --poisson_dir $(POISSON_DIR) --read_write_dir $(READWRITE_DIR)

compute_runtime :
	python $(COMPUTE_RUNTIME_SCRIPT) $(ARGS)

compute_runtime_gputpu :
	python $(COMPUTE_RUNTIME_GPUTPU_SCRIPT) $(ARGS)

compute_corrcoef :
	python $(COMPUTE_CORRCOEF_SCRIPT) $(ARGS)

plot_accuracy_runtime: 
	python $(PLOT_ACCURACY_RUNTIME_SCRIPT) $(ARGS)