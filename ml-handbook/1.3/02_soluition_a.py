def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    best_num = nums[0]
    best_count = 1

    count = dict()
    for num in nums:
        if num not in count: 
            count[num] = 1
        else:
            count[num] += 1

        if count[num] > best_count:
            best_count = count[num]
            best_num = num    

    return best_num

print(most_frequent([1, 2, 3, 1, 2, 3]))