import os
import numpy as np
import pandas as pd
from docplex.mp.model import Model
from visualize import plot_instances
from solutionfeatures import generate_solution_features


def cplex_model(arcs, customers, Capacity, nodes, dist, data, path, instance_name):
    mdl = Model("CVRP")
    x = mdl.binary_var_dict(arcs, name="x")  # variable x, type: binary, dictionary
    u = mdl.continuous_var_dict(
        customers, ub=Capacity, name="u"
    )  # cummulative variable for every customers with upper count of vehicle capacity

    mdl.minimize(
        mdl.sum(dist[i, j] * x[i, j] for i, j in arcs)
    )  # objective function of CVRP: minimize the dist

    mdl.add_constraints(
        mdl.sum(x[i, j] for j in nodes if j != i) == 1 for i in customers
    )  # adding constraints to the model
    mdl.add_constraints(
        mdl.sum(x[i, j] for i in nodes if i != j) == 1 for j in customers
    )
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(x[i, j], u[i] + data.demand[j] == u[j])
        for i, j in arcs
        if i != 0 and j != 0
    )
    mdl.add_constraints(u[i] >= data.demand[i] for i in customers)
    mdl.parameters.timelimit = 15
    solution = mdl.solve(log_output=False)  # solving vrp using cplex
    # print(solution)  # printing the solution
    solution.solve_status
    active_arcs = [a for a in arcs if x[a].solution_value > 0.9]
    solution.export_as_sol(
        path=path,
        basename=f"solution_{instance_name}",
    )
    return active_arcs, solution


def solution_features(instance_name, path):
    data = pd.read_csv(
        f"{path}/data.csv",
        header=None,
        names=("cust_loc_x", "cust_loc_y", "demand"),
    )
    info = pd.read_csv(f"{path}/info.csv")
    n_cust = int(info.customer.to_string(index=False))
    Capacity = int(info.Capacity.to_string(index=False))
    customers = [i for i in range(1, n_cust + 1)]
    nodes = [0] + customers
    arcs = [
        (i, j) for i in nodes for j in nodes if i != j
    ]  # set of arcs without diagonals
    dist = {
        (i, j): np.hypot(
            data.cust_loc_x[i] - data.cust_loc_x[j],
            data.cust_loc_y[i] - data.cust_loc_y[j],
        )
        for i, j in arcs
    }  # using Euclidean function

    active_arcs, solution = cplex_model(
        arcs, customers, Capacity, nodes, dist, data, path, instance_name
    )

    # plot instances
    plot_instances(data, active_arcs, path, instance_name)

    (
        solver,
        avg_direct_dist_depot_cust,
        avg_dist_btw_routes,
        maximum_distance_per_route,
        depth_route,
        variance,
        nroutes,
        cost,
        optimality,
    ) = generate_solution_features(data, active_arcs, solution)

    return (
        instance_name,
        solver,
        avg_direct_dist_depot_cust,
        avg_dist_btw_routes,
        maximum_distance_per_route,
        depth_route,
        variance,
        nroutes,
        cost,
        optimality,
    )


def generate_features(instance_name, path):

    (
        name,
        solver,
        avg_direct_dist_depot_cust,
        avg_dist_btw_routes,
        maximum_distance_per_route,
        depth_route,
        variance,
        nroutes,
        cost,
        optimality,
    ) = solution_features(instance_name, path)
    s = {
        "S1": name,
        "S2": solver,
        "S3": avg_direct_dist_depot_cust,
        "S4": avg_dist_btw_routes,
        "S5": variance,
        "S6": maximum_distance_per_route,
        "S7": depth_route,
        "S8": nroutes,
        "S9": cost,
        "S10": optimality,
    }
    solution = pd.DataFrame(data=s, index=[0])
    solution.to_csv(
        "featuresdata/solutionfeatures/SolutionGA.csv",
        mode="a",
        index=False,
        header=False,
    )


directory = "C:/Users/sakshi.singh/OneDrive - LichtBlick SE/Documents/VRP + GA + ML/vrp/vrp/data/"
for instance_name in os.listdir(directory):
    path = os.path.join(directory, instance_name)
    generate_features(instance_name, path)
