
# Local application imports
from .utils import constants, print_successful

# Standard library imports
import os
import logging
log = logging.getLogger(constants.logger_name)

# Third party imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from colorama import init, Back, Fore, Style
init(autoreset=True)





#TODO: review the resize of the images because it will depend on the number of people and months
#TODO: a lot of code of plots are repeated, improve it

_CONFIGS = {

    "show_plots": False,
    "figsize": (16,8),
    "filename": "image.png",
    "cmap":"gist_heat_r"
}

_FOLDER = "examples/imgs/"
class Plot:
    def __init__(self, folder_location):
        #TODO: Currently it is not possible to add new location
        self.folder_location = _FOLDER #TODO: how to set folder_location in a class and still use in a decorator? (save_fig)
        self.create_folder() #TODO: is this a good a idea?

    def create_folder(self):
        try:
            os.mkdir(self.folder_location)
        except OSError:
            log.info("Folder %s already exists" % self.folder_location)
            # print(Back.YELLOW + Fore.BLACK + Style.BRIGHT + "Folder %s already exists." % self.folder_location)
        else:
            print("Folder %s was created succesfully." % self.folder_location)

    def save_fig(**param):
        """ Decorator to save figure
            https://stackoverflow.com/questions/53449782/using-a-decorator-to-save-matplotlib-graphs-saved-output-is-blank
        """
        def outer(func):
            def inner(*args, **kwargs):

                if 'filename' in kwargs.keys():
                    filename = kwargs['filename']
                    try:
                        plt.savefig(filename)
                        utils.print_successful("The file %s was saved successfully in %s" % (filename, _FOLDER))
                    except:
                        utils.print_successful("Error in saving the file %s in %s" % (filename, _FOLDER))
                if 'show' in param.keys() and param["show"]:
                    print('show')
                    plt.show()
                else:
                    pass
            return inner
        return outer

    # @save_fig()
    def plot_bar_grouped(
                        self, 
                        df, 
                        selected_columns, 
                        xlabel = None,
                        ylabel = None,
                        title = None,
                        figsize = _CONFIGS["figsize"],
                        filename = _CONFIGS["filename"]):
        """Creates a plot bar grouped by `selected_columns` of de dataframe `df`

        Parameters
        ----------
        df : DataFrame
            dataframe with the information to plot
        selected_columns : array
            array with the columns it must be grouped on
        xlabel : str, optional
            label of the x-axis, by default None
        ylabel : str, optional
            label of the y-axis, by default None
        title : str, optional
            title of the plot, by default None
        figsize : tuple, optional
            tuple with the dimensions of the file plot (width, height), by default _CONFIGS["figsize"]
        filename : str, optional
            name of the file that will be saved, it can be full address, by default _CONFIGS["filename"]
        """

        #TODO: in order to use the 'mean' of plot_bar_grouped should we create a new function?


        df = df.groupby(selected_columns)[selected_columns[0]].count().reset_index(name='count')
        df = df.pivot(index=selected_columns[0], columns=selected_columns[1], values='count')

        ax = df.plot(kind="bar", stacked=True, figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        # plt.show()

    # @save_fig()
    def plot_heatmap(
        self, 
        df, 
        rows, 
        columns, 
        calc_field, 
        aggfunc = np.size,
        xlabel = None,
        ylabel = None,
        title = None,
        figsize = _CONFIGS["figsize"],
        filename = _CONFIGS["filename"],
        cmap=_CONFIGS["cmap"],
        fmt = 'g'):
        # TODO: too much arguments, better this way or add kwargs? seaborn/pandas does that https://github.com/pandas-dev/pandas/blob/65090286608602ba6d67ca4a29cf0535156cd509/pandas/core/tools/datetimes.py#L555
        """ Creates a heatmap plot based on the `df` and the rows, columns and the calc_field 
        
        If chosen: 
            -rows: 'month'
            -columns: 'person`
            -calc_field: `words` 
            -aggfunc: np.mean
            This will create a heatmap of the average words sent (label) by person (y-axis) by month (x-axis) 
        Parameters
        ----------
        df : DataFrame
            dataframe with the information to plot
        rows : str
            dataframe column that will be interpreted as the x-axis
        columns : str
            dataframe column that will be interpreted as the y-axis
        calc_field : str
            dataframe column used to calculate the information

        aggfunc : numpy/lambda function #TODO: what is the real type of 'np.size' np.mean'
            numpy function to calculate
            e.g., np.size for the count of messages, np.mean for the average
        xlabel : str, optional
            label of the x-axis, by default None
        ylabel : str, optional
            label of the y-axis, by default None
        title : str, optional
            title of the plot, by default None
        figsize : tuple, optional
            tuple with the dimensions of the file plot (width, height), by default _CONFIGS["figsize"]
        filename : str, optional
            name of the file that will be saved, it can be full address, by default _CONFIGS["filename"]
        cmap : str, optional
            colormap of the plot, 
            you can choose the options from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            by default _CONFIGS["cmap"]
        fmt : str, optional
            format of how to show the labels of the plot, by default 'g'

        More info: https://seaborn.pydata.org/generated/seaborn.heatmap.html
        """ 

        df = df.pivot_table(values=calc_field,
                            index=rows,
                            columns=columns,
                            aggfunc=aggfunc).dropna(axis=0, how='all')

        plt.figure(figsize=figsize)
        ax = sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.folder_location + filename)
        # plt.show()

    # @save_fig()
    def plot_heatmap_time(
        self, 
        df, 
        aggfunc = np.size,
        xlabel = None,
        ylabel = None,
        title = None,
        figsize = (11,15),
        filename = _CONFIGS["filename"],
        cmap=_CONFIGS["cmap"],
        fmt = '.1f'):
        """ Create a specific heatmap with the time day of the week and time
        #TODO: Should repeat basically the same docstring of the function plot_heatmap?
        #TODO: create a way to re-use the plot_heatmap instead of creating a new one like this
        Parameters
        ----------
        df : DataFrame
            [description]
        aggfunc : [type], optional
            [description], by default np.size
        xlabel : [type], optional
            [description], by default None
        ylabel : [type], optional
            [description], by default None
        title : [type], optional
            [description], by default None
        figsize : tuple, optional
            [description], by default (11,15)
        filename : [type], optional
            [description], by default _CONFIGS["filename"]
        cmap : [type], optional
            [description], by default _CONFIGS["cmap"]
        fmt : str, optional
            [description], by default '.1f'
        """
        #reference https://dfrieds.com/data-visualizations/when-use-heatmaps.html

        #TODO: a plot with all users based on the time they most interact


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
        f, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df3, annot=True, fmt=fmt, 
                        linewidths=.5, ax=ax, xticklabels=labels, 
                        yticklabels=all_hours, cmap=cmap)
        # ax.axes.set_title("Message by day and hour", fontsize=24, y=1.01)
        ax.set(xlabel='Weekday', ylabel='Hour');
        plt.savefig(self.folder_location + filename)


if __name__ == "__main__":
    from .data import create_dataframe, transform_dataframe
 

    df = transform_dataframe(create_dataframe())
    # print(Back.GREEN + Fore.RED + Style.BRIGHT + "teste")
    # Plot.plot_bar_grouped(df=df, 
                    # selected_columns=["month", "person"], 
                    # xlabel='Date', 
                    # ylabel='Quantity', 
                    # title='Messages',
                    # filename="bar_message_sent_date_person.png")