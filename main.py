import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


df = pd.read_csv(
    "/Users/blackprince/Desktop/Projects/Python/stock_prediction/NVDA_yfinance_clean.csv"
)
# Information about the dataset
print(df.head())
print(df.shape)
print(df.describe())
print(df.isnull().sum())


def generate_closing_plot_graph(df):
    """This function uses a OHLC data set of a stock and generate a closing price graph

    Args: A dataframe
    """
    plt.figure(figsize=(15, 7))
    plt.plot(df["Close"])
    plt.title("Nvidia closing price", fontsize=15)
    plt.show()


def generate_distribution_plot(df):
    """This function creates a distrubiton of the open, high, low, close, volume columns
    Args : it takes the dataframe as a arguement"""

    features = df.columns.to_list()
    features.remove("Date")
    if len(features) > 6:
        return 0

    plt.figure(figsize=(15, 7))
    for i, col in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sb.histplot(df[col], kde=True)

    plt.tight_layout()
    plt.show()


def generate_box_plot(df):
    """Generate box plot of the dataset, to identify outliers
    Args: Only the dataframe
    """
    features = df.columns.to_list()
    features.remove("Date")
    if len(features) > 6:
        return 0

    plt.figure(figsize=(15, 7))
    for i, col in enumerate(features):
        plt.subplot(2, 3, i + 1)
        sb.boxplot(x=df[col])

    plt.show()


def process_date_info(df):
    """This function takes the dates and create day ,month and year columns
    Args: takes a dataframe as argument
    """
    splitted = df["Date"].str.split("-", expand=True)
    df["Day"] = splitted[2].astype("int")
    df["Month"] = splitted[1].astype("int")
    df["Year"] = splitted[0].astype("int")
    return df


def group_months_in_quarter(df):
    """Group months in quarters and plot a graph
    Args:
        df: Dataframe
    Return:
        The dataframe with
    """

    df = process_date_info(df)
    df["is_quarter_end"] = np.where(df["Month"] % 3 == 0, 1, 0)
    data_grouped = df.drop("Date", axis=1).groupby("Year").mean()
    plt.figure(figsize=(20, 8))
    for i, col in enumerate(["Open", "High", "Low", "Close"]):
        plt.subplot(2, 2, i + 1)
        data_grouped[col].plot.bar()
    plt.show()

    return df


def target_feature(df, pie):
    """Add features to help the model classfify
    Arg:
        Df: Dataframe
        pie: plot the pie chart
        heatmmap: pplot the heatmap

    Return:
        The dataframe with new columns
    """
    df["close-open"] = df["close"] - df["open"]
    df["low-high"] = df["Low"] - df["High"]
    df["target"] = np.where(df["close"].shift(-1) > df["close", 1, 0])

    if pie:
        plt.pie(df["target"].value_counts().values, labels=[0, 1], autopct="%1.1f%%")
        plt.show()

    return df
