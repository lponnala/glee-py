
# Links to R code from which this python version has been created
# https://github.com/lponnala/omics/blob/main/2020-07-06/Rscript.R
# https://github.com/lponnala/omics/blob/main/2020-07-06/glee-funcs.R

import pandas as pd
import glee

tag = ['wt-k1','k1-k6','wt-k6'][0]
data_file = f"https://raw.githubusercontent.com/lponnala/omics/main/2020-07-06/data_glee_{tag}.csv"
fitplots_file = f"verify/glee_{tag}_fitplots.png"
stnpvals_file = f"verify/glee_{tag}_stnpval.png"
output_file = f"verify/glee_{tag}_results.csv"

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

proteins = D.iloc[:,0]
dat = {'A': D.iloc[:,1:(1+nA)], 'B': D.iloc[:,(1+nA):(1+nA+nB)]}
model = glee.fit_model(dat, fit_type=fit_type, num_pts=num_pts, num_std=num_std)
glee.model_fit_plots(model, file=fitplots_file)
stn_pval = glee.calc_stn_pval(dat, model, num_iter)
glee.stn_pval_plots(stn_pval, file=stnpvals_file)
tab = pd.concat(stn_pval.values(), axis=1, keys=stn_pval.keys()).assign(protein_id=proteins)
tab.to_csv(output_file)

res = pd.read_csv(f"https://raw.githubusercontent.com/lponnala/omics/main/2020-07-06/glee_{tag}-results.csv")
df = pd.merge(tab, res, on='protein_id', how='outer')
assert not df.isna().any().any(), "tab+res: check merge"
print((df['p_value'] - df['pVal']).describe())
top = min(20, df.shape[0])
top_match = 100*sum(df.sort_values(by='p_value').head(top)['protein_id'].values == df.sort_values(by='pVal').head(top)['protein_id'].values)/top
print(f"match among top {top} = {round(top_match)}%")
