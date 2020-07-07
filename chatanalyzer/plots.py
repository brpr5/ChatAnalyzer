
# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Local application imports



#TODO: review the resize of the images because it will depend on the number of people and months
#TODO: a lot of code of plots are repeated, improve it

_CONFIGS = {
    "show_plots": False,
    "figsize": (16,8)
}

_FOLDER = "examples/imgs"
class Plot:
    def __init__(self, folder_location):
        self.folder_location = folder_location
        self.create_folder() #TODO: is this a good a idea?

    def create_folder(self):
        try:
            os.mkdir(self.folder_location)
        except OSError:
            print("Folder %s already exists." % self.folder_location)
        else:
            print("Folder %s was created succesfully." % self.folder_location)

    def save_fig(**param):
        """ Decorator to save figure
            https://stackoverflow.com/questions/53449782/using-a-decorator-to-save-matplotlib-graphs-saved-output-is-blank
        """
        def outer(func):
            def inner(*args, **kwargs):
                artist = func(*args)
                if 'filename' in param.keys():
                    print('filename')
                    plt.savefig(param['filename'])
                if 'show' in param.keys() and param["show"]:
                    print('show')
                    plt.show()
                else:
                    return artist
            return inner
        return outer

    # @save_fig(**{'filename':'teste.png', 'show':_CONFIGS["show_plots"]})
    def plot_bar_grouped(self, df, selected_columns, **kwargs):
        """[summary]

        Args:
            df ([type]): [description]
        """
        #TODO: in order to use the 'mean' of plot_bar_grouped should we create a new function?
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        title = kwargs.get('title', None)
        figsize = kwargs.get('figsize', _CONFIGS["figsize"])
        filename = kwargs.get('filename', 'image.png')

        df = df.groupby(selected_columns)[selected_columns[0]].count().reset_index(name='count')
        df = df.pivot(index=selected_columns[0], columns=selected_columns[1], values='count')

        ax = df.plot(kind="bar", stacked=True, figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.folder_location + filename)
        # plt.show()

    def plot_heatmap(self, df, rows, columns, calc_field, aggfunc, **kwargs):
        """[summary]

        Args:
            df ([type]): [description]
            rows ([type]): [description]
            columns ([type]): [description]
            calc_field ([type]): [description]
            aggfunc ([type]): [description]
        """
        cmap = kwargs.get('cmap', 'YlGn')
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        title = kwargs.get('title', None)
        figsize = kwargs.get('figsize', _CONFIGS["figsize"])
        filename = kwargs.get('filename', 'image.png')
        fmt = kwargs.get('fmt', 'g')

        df = pivot(df=df, 
                        rows=rows, 
                        columns=columns, 
                        calc_field=calc_field,
                        aggfunc=aggfunc)

        plt.figure(figsize=figsize)
        ax = sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.folder_location + filename)
        # plt.show()


    def plot_heatmap_time(self, df, aggfunc, **kwargs):
        #reference https://dfrieds.com/data-visualizations/when-use-heatmaps.html

        #TODO: a plot with all users based on the time they most interact
        cmap = kwargs.get('cmap', 'gist_heat_r')
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)
        title = kwargs.get('title', None)
        filename = kwargs.get('filename', 'image.png')

        df2 = pd.pivot_table(df[['day_name', 'hour', 'person']], index=['day_name', 'hour'], aggfunc=aggfunc)
        df3 = df2.unstack(level=0)
        labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df3 = df3.reindex(labels=labels, axis=1, level=1)

        am_hours = []
        pm_hours = []
        for hour in range(1,12):
            am_hours.append(str(hour) + " am")
            pm_hours.append(str(hour) + ' pm')
        all_hours = ["12 am"] + am_hours + ["12 pm"] + pm_hours

        sns.set_context("talk")
        f, ax = plt.subplots(figsize=(11,15))
        ax = sns.heatmap(df3, annot=True, fmt=".1f", 
                        linewidths=.5, ax=ax, xticklabels=labels, 
                        yticklabels=all_hours, cmap=cmap)
        # ax.axes.set_title("Message by day and hour", fontsize=24, y=1.01)
        ax.set(xlabel='Weekday', ylabel='Hour');
        plt.savefig(self.folder_location + filename)

def pivot(df, rows, columns, calc_field, aggfunc):
    """[summary]

    Args:
        df ([type]): [description]
        rows ([type]): [description]
        columns ([type]): [description]
        calc_field ([type]): [description]
        type ([type]): [description]
            opts: np.mean, np.size

    Returns:
        [type]: [description]
    """

    df_pivot = df.pivot_table(  values=calc_field,
                                index=rows,
                                columns=columns,
                                aggfunc=aggfunc).dropna(axis=0, how='all')

    return df_pivot

if __name__ == "__main__":
    from data import create_dataframe, transform_dataframe

    df = transform_dataframe(create_dataframe())