import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df):
    sns.countplot(x='Outcome', data=df)
    plt.title('Distribuição de Diagnóstico (0=Não Diabético, 1=Sim Diabético)')
    plt.show()

def plot_histograms(df):
    df.hist(bins=20, figsize=(15,10))
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.show()