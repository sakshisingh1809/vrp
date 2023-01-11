import pandas as pd

path = "C:/Users/sakshi.singh/OneDrive - LichtBlick SE/Documents/VRP + GA + ML/vrp/vrp/featuresdata/"

# Instance feature file
read_file = pd.read_csv("instancefeatures/Instance.csv")
read_file.to_excel("instancefeatures/Instances.xlsx", index=None)

# CPLEX file
read_file = pd.read_csv(path + "solutionfeatures/SolutionsCPLEX.csv")
read_file.to_excel(
    path + "solutionfeatures/SolutionsCPLEX.xlsx", index=None, header=None
)  # Solutions feature file

# Genetic algorithm file
read_file = pd.read_csv(path + "solutionfeatures/SolutionsGA.csv")
read_file.to_excel(
    path + "solutionfeatures/SolutionsGA.xlsx", index=None, header=None
)  # Solutions feature file
