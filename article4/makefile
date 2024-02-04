BASEDIR=/Users/nickm/thesis/metareview/WeakBaselinesMLPDE/article4
READWRITE_DIR=/Users/nickm/thesis/metareview/WeakBaselinesMLPDE/article4

POISSON_DIR=$(BASEDIR)
MAKE_SCRIPT=$(BASEDIR)/scripts/make.mk
PYTHONPATH=$(BASEDIR)/scripts:$(BASEDIR)/code

export BASEDIR READWRITE_DIR POISSON_DIR PYTHONPATH


compute_runtime:
	make -f $(MAKE_SCRIPT) compute_runtime

compute_runtime_gputpu:
	make -f $(MAKE_SCRIPT) compute_runtime_gputpu

compute_corrcoef:
	make -f $(MAKE_SCRIPT) compute_corrcoef

plot_accuracy_runtime:
	make -f $(MAKE_SCRIPT) plot_accuracy_runtime