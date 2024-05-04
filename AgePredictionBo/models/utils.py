import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


# S'ha d'anar ampliant a mesura que les necessitem
def calculate_metrics(X_test = None, y_test = None, y_pred = None,
                       model = None, metrics_type='MSE'):
    if metrics_type == 'MSE':
        return np.square(np.subtract(y_test, y_pred))
    elif metrics_type == 'R2':
        return model.score(X_test, y_test)
    


def plot_results(y_pred, y_test, error):
    # Plot y_pred - MSE
    vari = calculate_metrics(y_test=y_test, y_pred=y_pred, metrics_type='MSE')
    mse = [np.array(vari).mean()]*len(vari)
    sns.scatterplot(x=y_pred,y=vari)
    sns.lineplot(x=y_pred, y=mse, label = 'MSE mean')
    plt.title('y_pred - MSE')
    plt.xlabel('y_pred')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    # Plot y_pred - y_test
    sns.scatterplot(x=y_test,y=y_pred)
    sns.lineplot(x=y_test, y=y_test, label='Perfect fit')
    plt.title('y_test - y_pred')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.legend()
    plt.show()

    # Histograma sortides
    sns.histplot(y_pred)
    plt.title('Hist predictions')
    plt.xlabel('y_pred')
    plt.ylabel('Frequency')
    plt.show()

    sns.histplot(error)
    plt.title('Hist errors')
    plt.xlabel('y_test - y_pred')
    plt.ylabel('Frequency')
    plt.show()