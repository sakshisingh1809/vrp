import os
import math
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances


def compute_average_distance_between_customers(x):
    return np.round(np.mean(pdist(x, "euclidean")), 5)


def compute_pairwise_variance_between_customers(x):
    return np.round(np.var(pdist(x, "euclidean")), 5)


def compute_average_distance_between_customers_depot(x, y):
    return np.round(np.mean(pairwise_distances(x, y, "euclidean")), 5)


def compute_pairwise_variance_between_customers_depot(x, y):
    return np.round(np.var(pairwise_distances(x, y, "euclidean")), 5)


def instance_generator(n_cust, vehicle_capacity, set_name):
    customer = [
        i for i in range(1, n_cust + 1)
    ]  # list of customers from (1,...,n_cust)
    nodes = [0] + customer  # set of nodes from (0 Union N), where Node 0= Depot
    instance_name = f"vrp-n{n_cust}-c{vehicle_capacity}-s{set_name}"

    demand = {0: 0}  # 0 demand for Depot '0'
    demand.update(
        {i: np.random.randint(1, vehicle_capacity) for i in customer}
    )  # add depot demand at the beginning of the dictionary
    n_vehicle = math.ceil((sum(demand.values())) / vehicle_capacity)

    # customer location coordinates including depot position
    grid = 1000
    cust_loc_x = np.round(np.random.rand(len(nodes)) * grid, 4)
    cust_loc_y = np.round(np.random.rand(len(nodes)) * grid, 4)
    
    d1 = {
        "Cust_loc_x": cust_loc_x,
        "Cust_loc_y": cust_loc_y,
        "Demand": demand.values(),
    }
    dataset1 = pd.DataFrame(data=d1)

    d2 = {
        "customer": n_cust,
        "vehicle": n_vehicle,
        "Capacity": vehicle_capacity,
    }
    dataset2 = pd.DataFrame(data=d2, index=[0])

    subfolder = os.path.join(set_name, instance_name)
    sheet1 = os.path.join(subfolder, "data.csv")
    sheet2 = os.path.join(subfolder, "info.csv")
    if not os.path.exists(set_name):
        os.makedirs(set_name)

    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    dataset1.to_csv(sheet1, header=False, index=False)
    dataset2.to_csv(sheet2, index=False)

    listr = []
    for value in demand.values():
        listr.append(value)
    demand_std = np.round(np.std(listr), 5)

    X = np.column_stack((cust_loc_x[1::], cust_loc_y[1::]))
    Y = np.column_stack((cust_loc_x[0], cust_loc_y[0]))
    avg_cust = math.ceil(n_cust / n_vehicle)
    avg_dist_btw_cust = compute_average_distance_between_customers(X)
    var_btw_cust = compute_pairwise_variance_between_customers(X)
    avg_dist_btw_cust_depot = compute_average_distance_between_customers_depot(
        X,
        Y,
    )
    var_btw_cust_depot = compute_pairwise_variance_between_customers_depot(X, Y)

    return (
        instance_name,
        avg_cust,
        n_vehicle,
        avg_dist_btw_cust,
        demand_std,
        var_btw_cust,
        avg_dist_btw_cust_depot,
        var_btw_cust_depot,
    )
