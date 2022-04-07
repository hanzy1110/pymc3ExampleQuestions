# %%
'''
Cuadratic Regression model and approximation of missing Data
'''
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
#Read the data from the pandas df

df = pd.read_csv('tempData.tsv', delimiter='\t').dropna()
df_na = pd.read_csv('tempData.tsv', delimiter='\t')
keys = list(df.keys())

time = df[keys[0]].values
time_na = df_na[keys[0]].values
temperature = df[keys[1]].values
temperature_na = df_na[keys[1]].values
#Plot the data to eyeball the model

plt.scatter(time, temperature)
plt.savefig('scatter.jpg')
plt.close()
# %%
#Write the model!
with pm.Model() as CuadraticRegression:
    # Long tailed priors ensure an information balance
    slope = pm.HalfCauchy('slope', beta=2, shape=(2,))
    intercept = pm.HalfCauchy('intercept', beta=2)
    noise = pm.HalfNormal('noise', sigma=1)
    #Mean regression curve
    mean = -slope[0] * time**2 + slope[1] * time + intercept
    #mean = -slope[0] * time**2  + intercept
    nu = pm.HalfNormal('nu', sigma=1)

    #StudentT likelihood gives better results in non-normally distributed models and non linear ones
    likelihood = pm.StudentT('likelihood', nu=nu, mu=mean, sigma=noise,
                             observed=temperature,
                             #                           testval=temp_test
                             )
    # Inference button!
    trace = pm.sample(draws=4000, tune=2000)
    az.plot_trace(trace)
    #Traceplot for diagnostic
    plt.savefig('traceplot.jpg')
    plt.close()
    #Print the summary of the inference
    summary = az.summary(trace)
    print(summary)

# %%
#Plot the model to see if it fits...
time_eval = np.linspace(time.min(), time.max(), num=100)
with CuadraticRegression:
    pm.plot_posterior_predictive_glm(

        trace, eval=time_eval, lm=lambda x, sample: -sample['slope'][0]*x**2 + sample['slope'][1]*x + sample['intercept'], **{'label':'Prediction'})
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.scatter(time, temperature, c='green', label='Data')
    plt.legend()
    plt.savefig('posterior_predictive.jpg')
    plt.close()


#%%
#Distributions for the missing data
with CuadraticRegression:
    labels = []
    for t_miss in time_na[np.isnan(temperature_na)]:
        labels.append(f'Tmiss for time:{t_miss}')
        T_missing = pm.Deterministic(labels[-1], -slope[0]*t_miss+slope[1]*t_miss+intercept) 
    samples = pm.sample_posterior_predictive(trace, samples=2000, var_names=labels)    
for label in labels:
    plt.hist(samples[label])
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.savefig(f'{label}.jpg')
    plt.close()
#Distributions for the parameters
with CuadraticRegression:
    samples = pm.sample_posterior_predictive(trace=trace, samples=1000, var_names=['slope', 'intercept'])
for label, sample in samples.items():
    plt.hist(sample)
    plt.xlabel(f'{label}')
    plt.ylabel('Frequency')
    plt.savefig(f'Cuadratic {label}.jpg')
    plt.close()





