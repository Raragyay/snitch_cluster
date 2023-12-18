import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import numpy as np
import seaborn.objects as so

def plot_result(y_col, df):
    hue = list(df.reset_index()[["direction", "prec","impl"]].apply(tuple, axis=1))
    g = sns.lineplot(data=df, x="num_data_points", y=y_col, hue=hue, marker="o")
    g.set_xscale("log", base=2)
    g.set_xlabel("Number of elements (C x H x W), N=1")
    g.get_legend().set_title("Implementation")
    g.grid()
    return g