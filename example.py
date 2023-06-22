
# Links to R code
# https://github.com/lponnala/omics/blob/main/2020-07-06/Rscript.R
# https://github.com/lponnala/omics/blob/main/2020-07-06/glee-funcs.R

import pandas as pd
import glee

data_file = "data_glee_k1-k6.csv"
fitplots_file = "glee_k1-k6_fitplots.png"
nA = 2
nB = 2
fit_type = "cubic"
num_iter = 10000
num_digits = 4
num_pts = 20
num_std = 5

D = pd.read_csv(data_file)
assert not D.isna().any().any(), "data: missing values found"
assert D.shape[0] >= 5, "data: not enough rows"
assert D.shape[1] == (1+nA+nB), "data: incorrect number of columns"

dat = {'A': D.iloc[:,1:(1+nA)], 'B': D.iloc[:,(1+nA):(1+nA+nB)]}
model = glee.fit_model(dat, fit_type=fit_type, num_pts=num_pts, num_std=num_std)
glee.model_fit_plots(model, file=fitplots_file)

best_model = model[max(model, key=lambda k: model[k]['adjRsq'])]

# list final p-values in the same order as D
# TODO