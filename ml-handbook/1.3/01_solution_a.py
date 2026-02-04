import numpy as np

def construct_matrix(first_array, second_array):
    """
    Construct matrix from pair of arrays
    :param first_array: first array
    :param second_array: second array
    :return: constructed matrix    
    """
    N = len(first_array)
    result = np.empty((N, 2))
    for i in range(N):
        result[i, 0] = first_array[i]
        result[i, 1] = second_array[i]

    return result