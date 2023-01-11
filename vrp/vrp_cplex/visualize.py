import matplotlib.pyplot as plt


def plot_instances(data, active_arcs, path, instance_name):
    # plot solution and then extract features
    plt.scatter(data.cust_loc_x[1:], data.cust_loc_y[1:], c="b")
    for i, j in active_arcs:
        plt.plot(
            [data.cust_loc_x[i], data.cust_loc_x[j]],
            [data.cust_loc_y[i], data.cust_loc_y[j]],
            c="g",
            alpha=0.3,
        )
    plt.plot(data.cust_loc_x[0], data.cust_loc_y[0], c="r", marker="s")
    plt.title(f"OptimalSolution_{instance_name}")
    plt.savefig(f"{path}/OptimalSolution_{instance_name}.png")
    plt.clf()
    plt.close()


""" def plot_solution(df, active_arcs):
    plt.scatter(df["Cust_loc_x"], df["Cust_loc_y"], c="b")
    plt.title(f"Solution: {df.Instance_name[0]}")
    for i in df.Customer:
        plt.annotate(
            "$q_%d=%d$" % (i, df.Demand[i - 1]),
            (df.Cust_loc_x[i - 1] + 2, df.Cust_loc_y[i - 1]),
        )
    for i, j in active_arcs:
        plt.plot(
            [df.Cust_loc_x[i], df.Cust_loc_x[j]],
            [df.Cust_loc_y[i], df.Cust_loc_y[j]],
            c="g",
            alpha=0.3,
        )
    plt.plot(df.Depot_Cust_loc_x[0], df.Depot_Cust_loc_y[0], c="r", marker="s")
    plt.axis("equal") """
