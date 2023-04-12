import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ts(y_: pd.Series, x_: pd.Series, title_=None, save=False):
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