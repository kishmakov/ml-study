import numpy as np

def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    values, counts = np.unique(nums, return_counts=True)
    return values[np.argmax(counts)]

print(most_frequent([1, 2, 3, 2, 1, 2, 3]))