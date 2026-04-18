import pandas as pd
from rich.console import Console

console = Console()

def load_data(config):
    users_df = pd.read_csv(config["users_path"])
    console.print("Users data loaded.")
    items_df = pd.read_csv(config["items_path"])
    console.print("Banners data loaded.")
    interactions_df = pd.read_csv(config["interactions_path"])
    console.print("Interactions data loaded.")
    return users_df, items_df, interactions_df