import numpy as np

def add_id(data):
    length = len(data)
    #print("length:",length)
    node_id = np.array(range(length))
    data_with_id = np.column_stack((data, node_id))

    return data_with_id