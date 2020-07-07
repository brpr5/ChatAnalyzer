from .. import data

import sys

modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]
print(allmodules)
# from plots import *
# import data
df = transform_dataframe(create_dataframe())

plot_bar_grouped(df=df, 
                    selected_columns=["month", "person"], 
                    xlabel='Date', 
                    ylabel='Quantity', 
                    title='Messages',
                    filename="bar_message_sent_date_person.png")

plot_heatmap(df=df,  
            rows='person',
            columns='month',
            calc_field='message',
            aggfunc=np.size,
            xlabel="Date",
            ylabel="Person",
            title="Messages sent by person, month",
            filename="heatmap_message_sent_date_person.png")


df_only_media = df[df["is_media"]==True]
plot_bar_grouped(df_only_media, 
                selected_columns=["month", "person"],
                xlabel='Date', 
                ylabel='Quantity', 
                title='Medias shared by person, month',
                filename="bar_media_sent_date_person.png")

plot_heatmap(df=df_only_media,
            rows='person',
            columns='month', 
            calc_field='is_media',
            aggfunc=np.size,
            xlabel="Date",
            ylabel="Person",
            title="Media sent by person, month",
            filename="heatmap_media_sent_date_person.png")


plot_heatmap(df=df,  
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


plot_heatmap_time(df=df, 
                aggfunc='count', 
                filename="heatmap_time_count")

plot_heatmap_time(df=df, 
                aggfunc=lambda x: len(x.unique()), 
                filename="heatmap_time_unique_users")