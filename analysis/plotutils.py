import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_horizontal_bar(data: pd.DataFrame, x_label = "Objects predicted") -> None:
    plt.figure(figsize=(1,1))
    with sns.axes_style('white'):
        g = sns.catplot(y="scenario", x="count", data=data, aspect=2, kind="bar", hue="label", palette="muted", errorbar=None)
        g.set_xticklabels()
        g.set_xlabels(x_label)
        g.set_ylabels("")

        ax = g.facet_axis(0,0)
        for c in ax.containers:
            labels = [' {}'.format(round(v.get_width()), '') for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')

    plt.savefig("Graphs/object_prediction_counts.svg", dpi=1200)
    plt.show()