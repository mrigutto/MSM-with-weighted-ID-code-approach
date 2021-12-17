This file explains how to use the code used for my individual report.

The code consists of three parts:
1) Pre-selection
Here, both the Pre-selection+ method by Hartveld et al. (2018) and my own Weighted ID code approach are coded. There are two separate functions to create the binary vectors, but the minhash and LSH functions are the same for both approaches.
2) MSM
This is the MSM approach as in Van Bezu et al. (2015).
3) Evaluation
This part runs the Pre-selection and MSM code for all thresholds between 0.05 and 1, with five bootstraps per threshold. It exports the statistics averaged over the bootstraps to Excel.

To run this code, save TVs-all-merged3.json in the same file as code.py. Then, run code.py in Python.