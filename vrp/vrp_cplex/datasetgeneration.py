import numpy as np
import pandas as pd
from instancegeneration import instance_generator
from openpyxl import load_workbook


set_name = "1"
start = 10
end = 50
for i in range(1000):
    n_cust = np.random.randint(start, end)
    vehicle_capacity = np.random.randint(start + 10, start + end + 20)
    (
        instance_name,
        avg_cust,
        n_vehicle,
        avg_dist_btw_cust,
        demand_std,
        var_btw_cust,
        avg_dist_btw_cust_depot,
        var_btw_cust_depot,
    ) = instance_generator(n_cust, vehicle_capacity, set_name)

    d = {
        "I1": instance_name,
        "I2": n_cust,
        "I3": vehicle_capacity,
        "I4": n_vehicle,
        "I5": demand_std,
        "I6": avg_cust,
        "I7": avg_dist_btw_cust,
        "I8": avg_dist_btw_cust_depot,
        "I9": var_btw_cust,
        "I10": var_btw_cust_depot,
    }
    dataset = pd.DataFrame(data=d, index=[0])
    if i == 0:
        dataset.to_csv(
            "featuresdata/instancefeatures/Instances.csv", mode="a", index=False
        )
    else:
        dataset.to_csv(
            "featuresdata/instancefeatures/Instances.csv",
            mode="a",
            index=False,
            header=False,
        )
