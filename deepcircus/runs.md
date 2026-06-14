# Exact decision tree solvers for small inputs (of length 16)

## 0: Branching on every bit, don't look at target

1'378'079

## 1: Branching on lowest bit

489'231

## 2: Looking for best bit recursively

133'378

# Using LogReg to trying to approximate number of nodes

train_avg_nodes: 16.07
test_avg_nodes: 17.33
logreg_test_mae: 18.84
logreg_test_mse: 3704.62
baseline_test_mae: 25.99
baseline_test_mse: 3237.50