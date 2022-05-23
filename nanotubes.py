'''
LinearRegression Model and estimation of missing data
'''
# %%
import os
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as tt

base = os.path.join('questions', 'second')
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
nsubs, nages = 27, 4
# draws = 25000
draws = 3500
tune=2000
#Reading the data from a tsv file
df = pd.read_csv('nanowire.csv').dropna()
x = df['x'].values
y = df['y'].values
#Plot the data to eyeball the model
plt.scatter(x,y, label='Thickness vs. Density')
plt.xlabel('Thickness')
plt.ylabel('Density')
plt.savefig(os.path.join(base,'plots', 'xvsy.jpg'))
keys = list(df.keys())

# %%
#Write the model!
with pm.Model() as NanoWires:
   # Mean regression line
    thetas = pm.Lognormal(r'$\theta_i$', mu=1, sigma=100, shape=(3,))
    theta_2 = pm.Uniform(r'$\theta_2$', lower=0, upper=1)
    mu = pm.Deterministic(r'$\mu$',thetas[0]*tt.exp(-theta_2*x**2)+thetas[1]*(1-tt.exp(-theta_2*x**2))*pm.invprobit(-x/thetas[-1]))
    likelihood = pm.Poisson('likelihood',
                           mu=mu,
                           observed=y,
                           )
    # Inference button
    # step = pm.Metropolis()
    trace = pm.sample(
                    # step=step,
                      draws=draws,
                      tune=tune,
                      # target_accept=0.98
                      )
    path = os.path.join(base, 'traces', 'trace.csv')
    df = pm.backends.tracetab.trace_to_dataframe(trace).to_csv(path)
    # Traceplot for diagnostic
    ax = az.plot_trace(trace, backend_kwargs={'figsize':(15,20)})
    ax.legend()
    path = os.path.join(base, 'plots', 'traceplot.jpg')
    plt.savefig(path)
    plt.close()
    # print the inference stats
    summary = az.summary(trace)
    path = os.path.join(base, 'summs', 'summ.csv')
    summary.to_csv(path)
    print(summary)

# %%
with NanoWires:
    mu = pm.Deterministic(r'$\mu(x=2)$',thetas[0]*tt.exp(-theta_2*2**2)+thetas[1]*(1-tt.exp(-theta_2*2**2))*pm.invprobit(-2/thetas[-1]))
    samples = pm.sample_posterior_predictive(trace=trace,
                                             samples=10000,
                                             var_names=[r'$\mu(x=2)$'])
# %%
pd.DataFrame().from_dict(samples).to_csv(os.path.join(base,'posterior', 'rho.csv'))
plt.style.use(['science', 'ieee'])
fig,ax = plt.subplots(1,1, figsize=(8,8))
ax.hist(samples[r'$\mu(x=2)$'])
ax.set_xlabel(r'$\mu(x=2)$')
ax.set_ylabel('Frequency')
plt.savefig(os.path.join(base, 'plots', 'intraClass.jpg'))
plt.close()

