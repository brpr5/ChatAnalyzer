import re
import pickle
from datetime import date
import string
import random
import os
import sys
from tkinter import filedialog, Tk
from math import ceil
# Third party imports
import pandas as pd
import numpy as np
import emoji

# Local application imports



#TODO: create a module for each based on a common class (Inheritance)
RE_FORMATS = {
    "all_message": r"(?P<all_message>\d{2}\/\d{2}\/\d{2}\d{2}.*?)(?=\d{2}\/\d{2}\/\d{2}\d{2})",
    "whatsapp": {
            "text": r"(?P<date>[0-9\/]+\s[0-9:]+)\s-\s(?P<person>[^:]+):\s(?P<message>.+)",
            "internal_message": r"((?P<date>[0-9\/]+\s[0-9:]+)\s-\s)?((?P<message>.*))",
            "media_not_uploaded": '<Arquivo de mídia oculto>',
            "to_rmv":[r"^(O histórico de conversas foi anexado ao e-mail como um arquivo.+)$"],
            "file_name":r"‎(?P<file_name>[\S]*)\s(?=\(arquivo anexado\))",
            "date":r"\[?([\d/-]+\s?[\d:]+)\]?\s[-]?\s?"
    },
    "telegram": {
    },
}

# For the Emoji analysis
SKIN_COLORS = [value for key, value in emoji.EMOJI_ALIAS_UNICODE.items() if "fitzpatrick" in key]
GENDER = ["♂", "♀"]


def get_file(file_path=None):

    if file_path:
        return file_path

    root = Tk()
    root.withdraw()
    file = filedialog.askopenfile(
        filetypes=[("Text file", "*.txt"), ("Zip file", "*.zip"), ("Comma-separated values", "*.csv")],
        initialdir=os.getcwd(),
        title='Choose a file')

    if not file:
        sys.exit(0)


    file_path = os.path.abspath(file.name)
    # file_name = os.path.basename(file.name)
    return file_path

def scramble_text_maintain_order(df, column_name="person"):
    """ This will change all the values for a scrambled one but it will maintain the same reference.

    Args:
        df (DataFrame): dataframe with the information for all messages with a specific column
        column_name (str, optional): [description]. Defaults to "person".
    """
    list_names = df[column_name].unique()

    dict_names = {name: scramble_text(name) for name in list_names}
    for key, value in dict_names.items():
        df.loc[df[column_name] == key, column_name] = value


def scramble_text(txt):
    """ It changes completely the meaning of a string in order to lose its readability

    Args:
        msg (str):   a string message that will change accordingly if is lowercase / uppercase and number.
                        Punctuation and emojis remains the same.

    Returns:
        [str]: the string with its chararacters scrambled.
    """

    if not txt:
        return None

    return "".join([random.choice(string.ascii_lowercase) if letter in string.ascii_lowercase
                    else random.choice(string.ascii_uppercase) if letter in string.ascii_uppercase
                    else random.choice(string.digits) if letter in string.digits
                    else letter
                    for letter in txt])


def get_emojis(txt):
    """ Will return only the emojis in the text
    This will return the base emoji, color and gender separated.
    #TODO: is there a way to maintain the color and gender of a emoji in a array?

    Args:
        txt (str): text that will be retrieved the emojis in it

    Returns:
        array: array with all emojis
    """
    if not txt:
        return None

    return [word for word in txt if word in emoji.UNICODE_EMOJI]


def create_dataframe(file_path=None):
    if not file_path:
        file_path = get_file(file_path)
    """Get a file Path of the file with the conversation and create a dataframe

    Args:
        file_path (str, optional): address to the file with the conversation. Defaults to "full.txt".

    Returns:
        DataFrame: raw dataframe of the conversation
    """
    # TODO: create an interface to choose the file, e.g. Tkinter
    # TODO: the pattern should be able to identify when a message is forward

    pat_all_message = re.compile(RE_FORMATS["all_message"], re.S | re.M)

    with open(file_path, "r") as f:
        contents = [m.group("all_message").strip().replace('\n', ' ')
                    for m in pat_all_message.finditer(f.read())]

    entry_columns = ["date", "person", "message"]
    values = []
    pat_entries = re.compile(RE_FORMATS["whatsapp"]["text"])
    for line in contents:
        values.append([m.group(var) for var in entry_columns for m in pat_entries.finditer(line)])

    df = pd.DataFrame(values, columns=entry_columns)

    return df


def transform_dataframe(df, scramble=True):
    """This will transform the dataframe with the information wanted. E.g., hour, emojis,  size etc

    Args:
        df ([DataFrame]): raw dataframe
        scramble (bool, optional):  if False it will not change the information of the content
                                    else it will change every letter and number. Defaults to True.

    Returns:
        DataFrame: dataframe with the information transformed
    """

    # TODO: date may be necessary to identify by itself because depending on the language it will be different
    df["month"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M").dt.to_period("M")
    df['date'] = pd.to_datetime(df['date'])
    # TODO is_media will be different based on the language
    df["is_media"] = df["message"].str.contains('<Arquivo de mídia oculto>')
    df["message"] = df["message"].str.lower()
    df["size_message"] = df["message"].str.len()
    df["words"] = df["message"].str.split().str.len()
    df["emojis"] = df["message"].apply(lambda row: get_emojis(row))
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour

    if(scramble):
        df['message'] = df['message'].apply(lambda x: scramble_text(x))
        scramble_text_maintain_order(df, 'person')
    return df


def save_dataframe(df, name="dataset"):
    """ This function will save the dataset in different ways.
    Currently it is only allows pickle

    Args:
        df (DataFrame): dataframe that will be saved
        name (str, optional): name of the file. Defaults to "dataset".
    """
    df.to_pickle(f"{name}.pkl")







if __name__ == "__main__":

    from plots import Plot
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    # plot = Plot("imgs/")

    df_created = create_dataframe("examples/_20200712.txt")
    # df_transformed = transform(dataframe_created, scramble_text=False)
    # df = transform_dataframe(create_dataframe("examples/_.txt"), scramble=False)
    print(df_created)
    # list_names = df["person"].dropna().unique()

    #TODO: create subplots when adding several person
