# %%
import os
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as tt
import seaborn as sns
plt.style.use(['science', 'ieee'])
base = os.path.join('questions', 'third')
paths = ['plots', 'traces', 'summs', 'posterior']
if not all(tuple(map(lambda p:os.path.exists(p), paths))):
    for path in paths:
        p=os.path.join(base, path)
        try:
            os.remove(p)
        except:
            pass
        os.makedirs(p,exist_ok=True)

# %%
draws = 25000
# draws = 6500
tune=3000
#Reading the data from a tsv file
#Building the design matrix X
df = pd.read_csv('insectData.tsv', delimiter='\t').dropna()
df = df.set_index(df['Board color'])
aux={}
trapped_insects = np.zeros(shape=(4,6))
for i,color in enumerate(df.index):
    vals = df.loc[color].values[1:]
    aux[color] = vals
    trapped_insects[i] = vals
trapped_insects_flattened = trapped_insects.flatten()
df_new = pd.DataFrame().from_dict(aux)
ax = sns.boxplot(data=df_new)
path = os.path.join(base, 'plots', 'boxplot.jpg')
plt.savefig(path)
plt.close()
X = np.zeros((trapped_insects_flattened.shape[0], 4))
s=0
for i,color in enumerate(df.index):
    X[s:s+6, i] = np.ones((6,))
    s+=6
# %%
#Write the model!
with pm.Model() as TrappedInsects:
    mean = pm.HalfCauchy("mean", beta=9, shape=(4,))

    sigma_b = pm.Gamma(r"$\sigma_b$", alpha=7.5, beta=1, shape=(4,))
    # sigma_b = pm.HalfNormal('sigma_b', sigma=1)
    treatments = pm.Normal("Treatments",
                           mu=mean,
                           sigma=sigma_b,
                           shape=(4,))

    computed = tt.dot(X, tt.transpose(treatments-mean))

    exp_error = pm.Gamma("Experimental Error", alpha=7, beta=1)
    # exp_error = pm.HalfNormal("exp_error", sigma=2)

    response = pm.Normal("Response",
                         mu=computed,
                         sd=exp_error,
                         observed=trapped_insects.flatten(),
                         # shape=(30,)
                         )

    trace = pm.sample(
                    # step=step,
                      draws=draws,
                      tune=tune,
                      target_accept=0.93
                      # target_accept=0.98
                      )
    path = os.path.join(base, 'traces', 'trace.csv')
    df = pm.backends.tracetab.trace_to_dataframe(trace).to_csv(path)
    # Traceplot for diagnostic
    ax = az.plot_trace(trace, backend_kwargs={'figsize':(15,20)})
    path = os.path.join(base, 'plots', 'traceplot.jpg')
    plt.savefig(path)
    plt.close()
    # print the inference stats
    summary = az.summary(trace)
    path = os.path.join(base, 'summs', 'summ.csv')
    summary.to_csv(path)
    print(summary)
    az.plot_posterior(trace, var_names=['Treatments'])
    path = os.path.join(base, 'plots', 'posterior_plot.jpg')
    plt.savefig(path)
    plt.close()
#%%
with TrappedInsects:
    for i, color in enumerate(df_new.keys()):
        _slice = np.delete(np.arange(4), i)
        contrast_to_control = pm.Deterministic(f"Diff w/{color}", 
                                           treatments[_slice] - treatments[i])
    samples = pm.sample_posterior_predictive(trace=trace,
                                             samples=5000,
                                             var_names=[f"Diff w/{color}" for color in df_new.keys()])
# %%
sns.set(rc={"figure.figsize":(15, 9)}) #width=3, #height=4
for key in samples.keys():
    samples[key] = samples[key].flatten()
sample_df = pd.DataFrame().from_dict(samples)
sns.boxplot(data=sample_df)
path = os.path.join(base, 'plots', 'differencePlot.jpg')
plt.savefig(path)
plt.close()
