import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataexploration import plot_distribution
from dataexploration import getbasicinfo


def loadexceldata(path, filename):
    instances = pd.read_excel(path + "instancefeatures/Instance.xlsx")
    solutionscplex = pd.read_excel(path + "solutionfeatures/SolutionsCPLEX.xlsx")
    solutionsga = pd.read_excel(path + "solutionfeatures/" + filename)

    """ getbasicinfo(solutionscplex)
    plot_distribution(solutionscplex, "CPLEX")
    getbasicinfo(solutionsga)
    plot_distribution(solutionsga, "GA") """

    if filename == "solution1.xlsx" or filename == "SolutionsGA.xlsx":
        sol = pd.concat([solutionscplex, solutionsga], axis=0)
        df = pd.merge(instances, sol, on="name")
    else:
        df = pd.merge(instances, solutionsga, on="name")
    # print(sol.shape, df.shape)
    return df
