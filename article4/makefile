BASEDIR=/Users/nickm/thesis/metareview/pde_runtime_comparisons/2d_navier_stokes
READWRITE_DIR=/Users/nickm/thesis/metareview/pde_runtime_comparisons/2d_navier_stokes

POISSON_DIR=$(BASEDIR)
MAKE_SCRIPT=$(BASEDIR)/scripts/make.mk
PYTHONPATH=$(BASEDIR)/scripts:$(BASEDIR)/code

export BASEDIR READWRITE_DIR POISSON_DIR PYTHONPATH


compute_runtime:
	make -f $(MAKE_SCRIPT) compute_runtime

compute_corrcoef:
	make -f $(MAKE_SCRIPT) compute_corrcoef

plot_accuracy_runtime:
	make -f $(MAKE_SCRIPT) plot_accuracy_runtime

print_FNO_statistics:
	make -f $(MAKE_SCRIPT) print_fno_statistics

print_CNO_statistics:
	make -f $(MAKE_SCRIPT) print_cno_statistics