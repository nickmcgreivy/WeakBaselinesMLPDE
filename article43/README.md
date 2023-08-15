## Instructions for running code

Simply type `delara_replicatetable1.py` and wait 10-20 minutes.

This produces plots identical to `replicate_table1.png` and `DOF_table1.png`.

## Instructions for interpreting results

We've replicated the test case described in the first paragraph of section 4, whose results are listed in table 1. In each case, the maximum error is between 6e-3 and 9e-3, and the low-order method takes 0.22-0.56s to run. We see that at the same error level, the 1D DG methods with different polynomial orders `p` each take about 0.05s. Thus, a stronger baseline outperforms the ML methods.