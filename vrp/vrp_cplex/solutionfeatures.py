import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances


def solution_routes(active_arcs):
    for i in active_arcs:
        starting_points = []
        for j in active_arcs:
            if j[0] == 0:
                starting_points.append(j)
    routes = []
    nroutes = 0
    for i in starting_points:
        subroutes = []
        for j in active_arcs:
            if i[1] == j[0]:
                # print(i[0], i[1], "->", j[0], j[1])
                subroutes.append(i)
                i = j
        subroutes.append(i)
        routes.append(subroutes)
        nroutes += 1

    return routes, nroutes


def compute_average_distance_between_customers(x):
    return np.round(np.mean(pdist(x, "euclidean")), 5)


def compute_pairwise_variance_between_customers(x):
    return np.round(np.var(pdist(x, "euclidean")), 5)


def centre_of_gravity(data, depot, route):
    total = depot.flatten()
    x = total[0]
    y = total[1]

    for j in route:
        x += data.cust_loc_x[j[0]]
        y += data.cust_loc_y[j[1]]
    x = np.round((x / len(route)), 5)
    y = np.round((y / len(route)), 5)
    center = [x, y]
    return center


def longest_distance(data, route):
    max = 0
    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]
        x = np.stack((data.cust_loc_x[node1[0]], data.cust_loc_y[node1[1]]))
        y = np.stack((data.cust_loc_x[node2[0]], data.cust_loc_y[node2[1]]))
        distance = np.round(
            pairwise_distances(x.reshape(1, -1), y.reshape(1, -1), "euclidean"), 5
        )
        if distance > max:
            max = distance

    maximum_distance_per_route = (max.flatten())[0]
    return maximum_distance_per_route


def depth(data, depot, route):
    max = 0
    for i in range(len(route)):
        node = np.stack((data.cust_loc_x[route[i][0]], data.cust_loc_y[route[i][1]]))
        distance = np.round(
            pairwise_distances(node.reshape(1, -1), depot, "euclidean"), 5
        )
        if distance > max:
            max = distance
            # j = route[i][1]
        # print(route[i], distance)
    return (max.flatten())[0]


def generate_solution_features(data, active_arcs, solution):
    # creating solution features
    depot = np.column_stack((data.cust_loc_x[0], data.cust_loc_y[0]))
    # cust = np.column_stack((data.cust_loc_x[1::], data.cust_loc_y[1::]))

    cust_direct_depot = np.stack(
        (data.cust_loc_x[j], data.cust_loc_y[j]) for i, j in active_arcs if i == 0
    )  # customers directly connected to the depot
    avg_direct_dist_depot_cust = compute_average_distance_between_customers(
        cust_direct_depot
    )
    routes, nroutes = solution_routes(active_arcs)

    list_of_centeroids = []
    variance = 0
    distance = 0
    depth_distance = 0
    maximum_distance_per_route = 0
    for i in range(nroutes - 1):
        route = routes[i]
        center = centre_of_gravity(data, depot, route)
        list_of_centeroids.append(center)
        variance += compute_pairwise_variance_between_customers(routes[i])
        distance = longest_distance(data, route)
        if distance > maximum_distance_per_route:
            maximum_distance_per_route = distance
        depth_distance += depth(data, depot, route)

    variance = np.round(variance, 5)
    avg_dist_btw_routes = compute_average_distance_between_customers(list_of_centeroids)
    maximum_distance_per_route = np.round(maximum_distance_per_route, 5)
    depth_route = np.round((depth_distance / nroutes), 5)

    cost = 0
    optimality = 0
    solver = "Exact"
    if solution.is_valid_solution:
        optimality = 1
        if solution.has_objective:
            cost = np.round(solution.objective_value, 5)

    return (
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
