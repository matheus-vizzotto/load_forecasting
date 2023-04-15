import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ts(x_: pd.Series, y_: pd.Series, title_=None, save=False):
    """_summary_

    Args:
        y_ (pd.Series): valores
        x_ (pd.Series): data
    """
    plt.figure(figsize=(15,5))
    sns.lineplot(x=x_, y=y_)
    plt.title(title_)
    if save:
        plt.savefig(f"../figs/{title_}.png")
    else:
        plt.show()

def plot_with_confidence(data: pd.DataFrame, x_: str, y_: str, levels: list=[90], title_=None, color: str='darkblue', alpha=.2, test: pd.Series=None, save=False):
    """Plota forecast junto com os intervalos de confiança. Importante: é necessário que os ICs estejam
    especificados no nome da coluna (ex.: ['AutoARIMA-lo-90', 'AutoARIMA-hi-90'])

    Args:
        y_ (pd.DataFrame): Dataframe com observado e ICs
        x_ (pd.Series): Data
        levels (list, optional): _description_. Defaults to [90].
        title_ (_type_, optional): _description_. Defaults to None.
        save (bool, optional): _description_. Defaults to False.
    """
    plt.figure(figsize=(15,5))
    plt.plot(data[x_], data[y_], color="black")
    if test:
        plt.scatter(data[x_], data[y_], color="black")
    for ic in levels:
        cols = [x for x in data.columns if f"{ic}" in x]
        print(cols)
        plt.fill_between(data[x_], data[cols[1]], data[cols[0]], alpha=0.2, label=f"IC -{ic}%", color=color)
    plt.title(title_)
    plt.legend(bbox_to_anchor = (1.15,1))
    plt.show()

