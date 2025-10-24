from enum import Enum
from typing import Optional
import logging
import pandas as pd

import config

def get_data_from_sheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Reads a public Google Sheet and returns its content as a pandas DataFrame.

    Args:
    sheet_id (str): The ID of the Google Sheet (can be found in the sheet's URL).
    sheet_name (str): The name of the sheet to read.

    Returns:
    pd.DataFrame: A DataFrame containing the sheet's data.
    """
    # Construct the URL for the CSV export of the sheet

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    return pd.read_csv(url)

class ModelType(Enum):
    GPT = ("gpt", "GPT")
    YA = ("ya", "YandexGPT")
    SBER = ("sber", "Sber")
    LOCAL = ("local", "Local")
    MISTRAL = ("mistral", "MistralAI")
    GGUF = ("gguf", "GGUF")

    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name


def format_df(df: pd.DataFrame, max_rows: Optional[int] = None, max_cols: Optional[int] = None) -> str:
    """
    Convert a pandas DataFrame to a formatted string suitable for Telegram.
    
    Args:
    df (pd.DataFrame): The DataFrame to format.
    max_rows (int, optional): Maximum number of rows to display. If None, display all rows.
    max_cols (int, optional): Maximum number of columns to display. If None, display all columns.
    
    Returns:
    str: A formatted string representation of the DataFrame.
    """
    # Limit the number of rows and columns if specified
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    if max_cols is not None and len(df.columns) > max_cols:
        df = df.iloc[:, :max_cols]

    # Convert DataFrame to string
    df_string = df.to_string(index = False)

    # Split the string into lines
    lines = df_string.split('\n')

    # Find the maximum length of any line
    max_length = max(len(line) for line in lines)

    # Create a top and bottom border
    border = '+' + '-' * (max_length + 2) + '+'

    # Add padding to each line and create the formatted string
    formatted_lines = [border]
    formatted_lines.extend(f"| {line:<{max_length}} |" for line in lines)
    formatted_lines.append(border)

    # Join the lines
    result = '\n'.join(formatted_lines)

    return f"```\n{result}\n```"


class UserManager:
    def __init__(self, sheet_id: str = config.USERS_SHEET_ID, sheet_name: str = 'users'):
        self.sheet_id=sheet_id
        self.sheet_name=sheet_name
        self.load_users()

    def load_users(self):
        self.users = get_data_from_sheet(self.sheet_id, self.sheet_name)

    def get_allowed_models(self, username: str) -> list:
        """
        Retrieve the list of models for a given username from the DataFrame.
        
        Args:
        username (str): The Telegram username to look up (with or without '@').
        df (pd.DataFrame): The DataFrame containing user data.
        
        Returns:
        list: A list of ModelType enums for the user, or an empty list if user not found.
        """
        # Ensure the username starts with '@'
        if not username.startswith('@'):
            username = f'@{username}'

        # Find the user in the DataFrame
        user_row = self.users[self.users['user'] == username]

        if user_row.empty:
            return []

        # Get the models string
        models_str = user_row['models'].iloc[0].strip()

        # If models is '*', return all ModelTypes
        if models_str == '*':
            return [model.value for model in ModelType]

        # Otherwise, split the string and convert to ModelTypes
        model_strings = models_str.split(',')
        models = []
        for model_str in model_strings:
            model_str = model_str.strip().lower()
            try:
                # Find the ModelType enum member with the matching value
                model = next(model for model in ModelType if model.value == model_str)
                models.append(model.value)
            except StopIteration:
                logging.warning(f"Warning: Unknown model type '{model_str}' for user {username}")

        return models
    
    def is_allowed(self, username: str) -> bool:
        if not username.startswith('@'):
            username = f'@{username}'
        return not self.users[self.users['user'] == username].empty

    def is_model_allowed(self, username: str, model: ModelType) -> bool:
        if not username.startswith('@'):
            username = f'@{username}'
        if self.is_allowed(username):
            return model.value in self.get_user_models(username)
        return False

    def is_admin(self, username: str) -> bool:
        if not username.startswith('@'):
            username = f'@{username}'
        if not self.users[self.users['user'] == username].empty:
            return self.users[self.users['user'] == username]['admin'].iloc[0].strip().lower() == 'yes'
        
    def get_role(self, username: str) -> bool:
        if not username.startswith('@'):
            username = f'@{username}'
        if not self.users[self.users['user'] == username].empty:
            return self.users[self.users['user'] == username]['role'].iloc[0].strip().lower()


    def get_users_string(self) -> str:
        return format_df(self.users)
        #return self.users.to_markdown(index=False)

