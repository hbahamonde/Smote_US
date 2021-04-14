import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from statsmodels.formula.api import ols
from scipy import stats

# load data file
nombres = ['R_lin','R_rbf','R_mlp','R_nb',
           'PPV_lin','PPV_rbf','PPV_mlp','PPV_nb',
           'fS_lin','fS_rbf','fS_mlp','fS_nb',
           'AUC_lin','AUC_rbf','AUC_mlp','AUC_nb']
resultados = []
for n in nombres:
    df = pd.read_csv(n+"3.csv", sep=",")
    dfv1 = df.to_numpy().ravel()
    df = pd.read_csv(n+"4.csv", sep=",")
    dfv2 = df.to_numpy().ravel()
    df = pd.read_csv(n+"5.csv", sep=",")
    dfv3 = df.to_numpy().ravel()
    df = pd.read_csv(n+"10.csv", sep=",")
    dfv4 = df.to_numpy().ravel()
    lista = list(zip(dfv1,dfv2,dfv3,dfv4))
    df = pd.DataFrame(lista,columns=['3 folds','4 folds','5 folds','10 folds'])

    df_melt = pd.melt(df.reset_index(), id_vars=['index'],
                      value_vars=['3 folds','4 folds','5 folds','10 folds'])
    df_melt.columns = ['index', 'variables', 'value']

    # generate a boxplot to see the data distribution by treatments. Using boxplot,
    # we can easily detect the differences between different number of folds
    ax = sns.boxplot(x='variables', y='value', data=df_melt, color='#99c2a2')
    ax = sns.swarmplot(x="variables", y="value", data=df_melt, color='#7d0013')
    ax.set_xlabel('Nbr of folds')
    ax.set_ylabel('Score')
    plt.savefig(n+'_histogram.eps',transparent=True)

    # anova table
    model = ols('value ~ C(variables)', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    resultados.append(anova_table['PR(>F)'][0])

np.savetxt("ANOVA(F-test).csv",resultados, delimiter=",")

nombres = ['R','PPV','fS','AUC']
k = 3
numero = '10'
df = pd.read_csv(nombres[k]+"_lin"+numero+".csv", sep=",")
dfv1 = df.to_numpy().ravel()
df = pd.read_csv(nombres[k]+"_rbf"+numero+".csv", sep=",")
dfv2 = df.to_numpy().ravel()
df = pd.read_csv(nombres[k]+"_mlp"+numero+".csv", sep=",")
dfv3 = df.to_numpy().ravel()
df = pd.read_csv(nombres[k]+"_nb"+numero+".csv", sep=",")
dfv4 = df.to_numpy().ravel()
lista = list(zip(dfv1,dfv2,dfv3,dfv4))
df = pd.DataFrame(lista,columns=['Lin','RBF','MLP','GNB'])

df_melt = pd.melt(df.reset_index(), id_vars=['index'],
                  value_vars=['Lin','RBF','MLP','GNB'])
df_melt.columns = ['index', 'methods', 'value']
# anova table
model = ols('value ~ C(methods)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(stats.ttest_ind(dfv2,dfv1, equal_var = False))
print(stats.ttest_ind(dfv2,dfv3, equal_var = False))
print(stats.ttest_ind(dfv2,dfv4, equal_var = False))
print(anova_table)


