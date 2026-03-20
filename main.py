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
    plt.figure(figsize=(12, 4))
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


generate_box_plot(df)
