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
import seaborn as sns
plt.style.use(['science', 'ieee'])

base = os.path.join('questions', 'first')
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
def get_df_data(df, nsubs, nages):
    subjects = df['Subject'].unique()
    ages = np.ndarray(shape=(nsubs, nages))
    sizes = np.ndarray(shape=(nsubs, nages))
    sex = np.ndarray(shape=(nsubs, nages))
    for i,sub in enumerate(subjects):
        ages[i] = df[df['Subject'] == sub]['age'].values
        sizes[i] = df[df['Subject'] == sub]['y'].values
        sex[i] = df[df['Subject'] == sub]['Sex_coded'].values
    return {'ages':ages, 'sizes':sizes, 'sex':sex}

nsubs, nages = 27, 4
draws = 5500
tune=2000
#Reading the data from a tsv file
df = pd.read_csv('ortho.csv').dropna()
#Plot the data to eyeball the model
df.hist()
plt.savefig('df_hist.jpg')
keys = list(df.keys())

data = get_df_data(df, nsubs, nages)
# %%
#Write the model!
with pm.Model() as OrthoDistance:
    betas = pm.Normal(r'$\beta_i$', mu=0, sigma=10e8, shape=(3,))
    tau_e = pm.Gamma(r'$\tau_e$', alpha=0.1, beta=0.1)
    tau_u = pm.Gamma(r'$\tau_u$', alpha=0.1, beta=0.1)
    u_i = pm.Normal(r'$u_i$', mu=0, tau=tau_u, shape=(nsubs,nages))
    # Mean regression line
    mean = betas[0] + betas[1]*data['ages'] + betas[2]*data['sex'] + u_i
    sigma_e = pm.HalfNormal(r'$\sigma_e$', tau=tau_e)
    sigma_u = pm.HalfNormal(r'$\sigma_u$', tau=tau_u)

    # StudentT likelihood gives better results for this data
    likelihood = pm.Normal('likelihood',
                           mu=mean,
                           sigma=sigma_e,
                           observed=data['sizes'],
                           )
    # Inference button
    trace = pm.sample(draws=draws,tune=tune, target_accept=0.93)
    path = os.path.join(base, 'traces', 'trace.csv')
    df = pm.backends.tracetab.trace_to_dataframe(trace).to_csv(path)
    # Traceplot for diagnostic
    az.plot_trace(trace, backend_kwargs={'figsize':(15,20)})

    path = os.path.join(base, 'plots', 'traceplot.jpg')
    plt.savefig(path)
    plt.close()
    # print the inference stats
    summary = az.summary(trace)
    path = os.path.join(base, 'summs', 'summ.csv')
    summary.to_csv(path)
    print(summary)
    az.plot_posterior(trace, var_names=[r'$\beta_i$',r'$\sigma_e$',r'$\sigma_u$'])
    path = os.path.join(base, 'plots', 'posterior_plot.jpg')
    plt.savefig(path)
    plt.close()
# %%
with OrthoDistance:
    rho = pm.Deterministic(r'$\rho$', (1/tau_u)**2/(sigma_e**2+(1/tau_u)**2))
    # sigma_e = pm.Deterministic(r'$\sigma_e$',1/tau_e)
    # sigma_u = pm.Deterministic(r'$\sigma_u$',1/tau_u)
    samples = pm.sample_posterior_predictive(trace=trace,
                                             samples=10000,
                                             var_names=[r'$\rho$'
                                                        # r'$\sigma_e$',
                                                        # r'$\sigma_u$'
                                                        ])

pd.DataFrame().from_dict(samples).to_csv(os.path.join(base,'posterior', 'ppc.csv'))
fig,ax = plt.subplots(1,1, figsize=(8,8))
ax.hist(samples[r'$\rho$'])
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('Frequency')
plt.savefig(os.path.join(base, 'plots', 'intraClass.jpg'))
plt.close()

# fig,ax = plt.subplots(1,2, figsize=(8,8))
# ax[0].hist(samples[r'$\sigma_u$'])
# ax[0].set_xlabel(r'$\sigma_u$')
# ax[0].set_ylabel('Frequency')
# ax[1].hist(samples[r'$\sigma_e$'])
# ax[1].set_xlabel(r'$\sigma_e$')
# ax[1].set_ylabel('Frequency')
# plt.savefig(os.path.join(base, 'plots', 'sigmas.jpg'))
# plt.close()

print('-x-'*30)
#No random Effect model
with pm.Model() as OrthoDistanceNoU:
    betas = pm.Normal(r'$\beta_i$', mu=0, sigma=10e8, shape=(3,))
    tau_e = pm.Gamma(r'$\tau_e$', alpha=0.1, beta=0.1)
    # Mean regression line
    mean = betas[0] + betas[1]*data['ages'] + betas[2]*data['sex']
    sigma_e = pm.HalfNormal(r'$\sigma_e$', tau=tau_e)

    # StudentT likelihood gives better results for this data
    likelihood = pm.Normal('likelihood',
                           mu=mean,
                           sigma=sigma_e,
                           observed=data['sizes'],
                           )
    # Inference button
    trace = pm.sample(draws = draws,tune=tune, target_accept=0.93)
    path = os.path.join(base, 'traces', 'trace_noU.csv')
    df = pm.backends.tracetab.trace_to_dataframe(trace).to_csv(path)
    # Traceplot for diagnostic
    az.plot_trace(trace, backend_kwargs={'figsize':(15,20)})

    path = os.path.join(base, 'plots', 'traceplot_NoU.jpg')
    plt.savefig(path)
    plt.close()
    # print the inference stats
    summary = az.summary(trace)
    path = os.path.join(base, 'summs', 'summ_noU.csv')
    summary.to_csv(path)
    print(summary)
    az.plot_posterior(trace)
    path = os.path.join(base, 'plots', 'posterior_plotNoU.jpg')
    plt.savefig(path)
    plt.close()
