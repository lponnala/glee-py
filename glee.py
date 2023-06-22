
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def fit_model(dat, fit_type, num_pts=20, num_std=5):
    model = {}
    for k in dat:
        xbar = dat[k].mean(axis=1)
        stdev = dat[k].std(axis=1)
        ind = (xbar > 0) & (stdev > 0)
        xbar_log = np.log(xbar.loc[ind])
        stdev_log = np.log(stdev.loc[ind])
        # fit the specified type of model
        df = pd.concat([xbar_log.rename('xbar_log'), stdev_log.rename('stdev_log')], axis=1)
        if fit_type == 'linear':
            reg = smf.ols(formula='stdev_log ~ 1 + xbar_log', data=df).fit()
        elif fit_type == 'cubic':
            reg = smf.ols(formula='stdev_log ~ 1 + xbar_log + I(xbar_log**2) + I(xbar_log**3)', data=df).fit()
        coeffs = reg.params
        adjRsq = reg.rsquared_adj
        # pick a few values to show model fitted values
        xbar_showfit = pd.Series(np.linspace(start=xbar_log.min(), stop=xbar_log.max(), num=num_pts)).rename('xbar_log')
        stdev_showfit = reg.predict(xbar_showfit.to_frame()).rename('stdev_log')
        if fit_type == 'linear':
            stdev_manualc = coeffs.loc['Intercept'] + coeffs.loc['xbar_log']*xbar_showfit
            assert np.allclose(stdev_showfit, stdev_manualc), "stdev_showfit: mismatch"
        elif fit_type == 'cubic':
            stdev_manualc = coeffs.loc['Intercept'] + coeffs.loc['xbar_log']*xbar_showfit + coeffs.loc['I(xbar_log ** 2)']*(xbar_showfit**2) + coeffs.loc['I(xbar_log ** 3)']*(xbar_showfit**3)
            assert np.allclose(stdev_showfit, stdev_manualc), "stdev_showfit: mismatch"
        # remove outliers, i.e. keep points that are within a few "sigma" of stdev
        ind_show = (stdev_log > (stdev_log.mean() - num_std*stdev_log.std())) & (stdev_log < (stdev_log.mean() + num_std*stdev_log.std()))
        xbar_showpts = xbar_log[ind_show].rename('xbar_log')
        stdev_showpts = stdev_log[ind_show].rename('stdev_log')
        # collect into a dict to be returned
        model[k] = {'model': reg, 'adjRsq': adjRsq, 'coeffs': coeffs, 'xbar_showfit': xbar_showfit, 'stdev_showfit': stdev_showfit, 'xbar_showpts': xbar_showpts, 'stdev_showpts': stdev_showpts, 'xbar': xbar, 'stdev': stdev}
    return model

def model_fit_plots(model, file=None):
    _,axs = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
    for i,k in enumerate(model):
        mod = model[k]
        df_pts = pd.concat([mod['xbar_showpts'], mod['stdev_showpts']], axis=1)
        df_pts.plot(kind='scatter', x='xbar_log', y='stdev_log', grid=True, ax=axs[i,0], alpha=0.3)
        df_fit = pd.concat([mod['xbar_showfit'], mod['stdev_showfit']], axis=1)
        df_fit.set_index('xbar_log').squeeze().plot(kind='line', xlabel='log(mean)', ylabel='log(stdev)', title=f"Condition {k}", grid=True, ax=axs[i,0])
        axs[i,0].text(x=df_fit['xbar_log'].min(), y=df_fit['stdev_log'].max(), s=f"adj.Rsq = {round(mod['adjRsq'],3)}")
        df_pts['xbar_log'].sort_values().to_frame().assign(protein=list(range(1,df_pts.shape[0]+1))).plot(kind='scatter', x='protein', y='xbar_log', ylabel='signal level (mean)', title=f"Condition {k}", grid=True, ax=axs[i,1])
    plt.tight_layout()
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
    plt.close()


# ---- OLD CODE ----

# import statsmodels.api as sm
# reg = sm.OLS(stdev_log, sm.add_constant(xbar_log.rename('x'))).fit()
# stdev_showfit = reg.predict(sm.add_constant(xbar_showfit)) # reg.predict(pd.concat([pd.Series([1]*num_pts).rename('const'), pd.Series(xbar_showfit).rename('x')], axis=1))
# assert np.allclose(stdev_showfit, coeffs.loc['const'] + coeffs.loc['x']*xbar_showfit), "stdev_showfit: mismatch"
