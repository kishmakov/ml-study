import numpy as np

def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    return np.bincount(nums).argmax()

print(most_frequent([1, 2, 3, 3, 1, 2, 3]))