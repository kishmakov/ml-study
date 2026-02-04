import numpy as np

def construct_matrix(first_array, second_array):
    """
    Construct matrix from pair of arrays
    :param first_array: first array
    :param second_array: second array
    :return: constructed matrix    
    """
    return np.column_stack((first_array, second_array))