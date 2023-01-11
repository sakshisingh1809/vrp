import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_distribution(df, title):

    plt.figure(figsize=(24, 20))

    plt.subplot(4, 2, 1)
    fig = df["S3"].hist(bins=20)
    fig.set_xlabel("Average distance between depot to directly-connected customers")
    fig.set_ylabel("Solutions")

    plt.subplot(4, 2, 2)
    fig = df["S4"].hist(bins=20)
    fig.set_xlabel("Average distance between routes")
    fig.set_ylabel("Solutions")

    plt.subplot(4, 2, 3)
    fig = df["S5"].hist(bins=20)
    fig.set_xlabel("Variance in number of customers per route")
    fig.set_ylabel("Solutions")

    plt.subplot(4, 2, 4)
    fig = df["S6"].hist(bins=20)
    fig.set_xlabel("Longest distance between two connected customers, per route")
    fig.set_ylabel("Solutions")

    plt.subplot(4, 2, 5)
    fig = df["S7"].hist(bins=20)
    fig.set_xlabel("Average depth per route")
    fig.set_ylabel("Solutions")

    plt.subplot(4, 2, 6)
    fig = df["S8"].hist(bins=20)
    fig.set_xlabel("Number of routes")
    fig.set_ylabel("Solutions")

    plt.suptitle(title)
    plt.show()


def getbasicinfo(df):
    print("Dataframe shape: ", df.shape)
    print("Dataframe: ", df.head())
    col_names = df.columns
    print("Column names: ", col_names)
    print(
        "Distribution of target: ", df["S10"].value_counts()
    )  # distribution of target
    print(
        "Percentage of distribution of target: ",
        (df["S10"].value_counts() / np.float64(len(df)) * 100),
    )  # percentage of distribution of target
    print("Dataframe info: ", df.info())
    print("Dataframe null values: ", df.isnull().sum())
