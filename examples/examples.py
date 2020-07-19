""" Examples of plots and use of functions

"""
# Local application imports
from chatanalyzer.data import transform_dataframe, create_dataframe
from chatanalyzer.plots import Plot

# Standard library imports
import os

# Third party imports
import numpy as np

df = transform_dataframe(create_dataframe())

plot = Plot("examples/imgs/")

plot.plot_bar_grouped(df=df, 
                    selected_columns=["month", "person"], 
                    xlabel='Date', 
                    ylabel='Quantity', 
                    title='Messages',
                    filename="bar_message_sent_date_person.png")

plot.plot_heatmap(df=df,  
            rows='person',
            columns='month',
            calc_field='message',
            aggfunc=np.size,
            xlabel="Date",
            ylabel="Person",
            title="Messages sent by person, month",
            filename="heatmap_message_sent_date_person.png")


df_only_media = df[df["is_media"]==True]
plot.plot_bar_grouped(df_only_media, 
                selected_columns=["month", "person"],
                xlabel='Date', 
                ylabel='Quantity', 
                title='Medias shared by person, month',
                filename="bar_media_sent_date_person.png")

plot.plot_heatmap(df=df_only_media,
            rows='person',
            columns='month', 
            calc_field='is_media',
            aggfunc=np.size,
            xlabel="Date",
            ylabel="Person",
            title="Media sent by person, month",
            filename="heatmap_media_sent_date_person.png")


plot.plot_heatmap(df=df,  
            rows='person',
            columns='month',
            calc_field='size_message',
            aggfunc=np.mean,
            xlabel="Date",
            ylabel="Person",
            title="Average Size Messages sent by person, month",
            figsize=(23,6),
            fmt='.1f',
            filename="heatmap_avg_message_sent_date_person.png")


plot.plot_heatmap_time(df=df, 
                aggfunc=np.size, 
                filename="heatmap_time_count")

plot.plot_heatmap_time(df=df, 
                aggfunc=lambda x: len(x.unique()), 
                filename="heatmap_time_unique_users")