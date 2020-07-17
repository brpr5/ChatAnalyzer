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
    """This will change all the values for a scrambled one but it will maintain the same reference.

    Parameters
    ----------
    df : DataFrame
        dataframe with the information for all messages with a specific column
    column_name : str, optional
        column that will be used to keep reference, by default "person"
    """

    list_names = df[column_name].unique()

    dict_names = {name: scramble_text(name) for name in list_names}
    for key, value in dict_names.items():
        df.loc[df[column_name] == key, column_name] = value


def scramble_text(txt):
    """Completely changes the meaning of a string in order lose its readability

    Parameters
    ----------
    txt : str
        message that will change accordingly if it is lowercase, uppercase or a number.
        Punctution and emojis remain the same

    Returns
    -------
    str
        string with its characters scrambled
    """   

    if not txt:
        return None

    return "".join([random.choice(string.ascii_lowercase) if letter in string.ascii_lowercase
                    else random.choice(string.ascii_uppercase) if letter in string.ascii_uppercase
                    else random.choice(string.digits) if letter in string.digits
                    else letter
                    for letter in txt])


def get_emojis(txt):
    """Get emojis in a given string and return a array with all emojis, color skins and gender

    Parameters
    ----------
    txt : str
        text to get emoji information

    Returns
    -------
    array
        array with base-emoji, color skins and gender
    """    

    #TODO: is there a way to maintain the color and gender of a emoji in a array?
    if not txt:
        return None

    return [word for word in txt if word in emoji.UNICODE_EMOJI]


def create_dataframe(file_path=None):
    """Create a dataframe based on the file path given

    Parameters
    ----------
    file_path : str, optional
        address to the file with the conversation, by default None

    Returns
    -------
    DataFrame
        raw dataframe of the conversation
    """
    # TODO: create an interface to choose the file, e.g. Tkinter
    # TODO: the pattern should be able to identify when a message is forward

    if not file_path:
        file_path = get_file(file_path)

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


def transform_dataframe(df, scramble=True, lowercase=False):
    """Transforms the DataFrame and add a few more columns based on the raw information

    Parameters
    ----------
    df : DataFrame
        raw DataFrame with the conversation information, base columns: date, person, message
    scramble : bool, optional
        if False it will not change the content information 
        else it will change every letter and number thus changing its meaning, by default True
    lowercase : bool
        if the message should be changed to lowercase, by default False
    Returns
    -------
    DataFrame
        a new dataframe with more columns added
    """    

    
    # TODO: is_media message will be different based on the language

    # TODO: date may be necessary to identify by itself because depending on the language it will be different
    # transform date information into datetime
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y %H:%M") #
    # 'month' information like 01-2020 for January, 2020
    df["month"] = df["date"].dt.to_period("M")
    # create colum that will show when a user have sent a file #TODO: distinguish between media with and without message
    df["is_media"] = df["message"].str.contains('<Arquivo de mídia oculto>')
    # get the size of the message
    df["size_message"] = df["message"].str.len()
    # number of the words sent by user #TODO: should remove emojis? 
    df["words"] = df["message"].str.split().str.len()
    # get an array of all emojis sent
    df["emojis"] = df["message"].apply(lambda row: get_emojis(row))
    # weekday name, e.g.: Sunday, Monday, etc
    df['day_name'] = df['date'].dt.day_name()
    # closed hour it was sent the message, e.g.: 1, 2, 3 o'clock
    df['hour'] = df['date'].dt.hour

    
    if lowercase:
        df["message"] = df["message"].str.lower()

    if scramble:
        df['message'] = df['message'].apply(lambda x: scramble_text(x))
        scramble_text_maintain_order(df, 'person')
    
    return df


def save_dataframe(df, name="dataset"):
    """Save dataset in different formats #TODO: pending add more formats

    Parameters
    ----------
    df : DataFrame
        dataframe that will be saved
    name : str, optional
        name of the file, by default "dataset"
    """    
    
    df.to_pickle(f"{name}.pkl")







if __name__ == "__main__":

    df_created = create_dataframe("examples/_20200712.txt")
    df_transformed = transform_dataframe(df_created, scramble=False, lowercase=True)

    print(df_created)


    #TODO: create subplots when adding several person
