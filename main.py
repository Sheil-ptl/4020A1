import numpy as np

#load the data

data = np.loadtxt('CIS4020F25DataSet.csv', delimiter=',', dtype=str)

#remove the identifiers, the ID's and convert the values to floats
data = data[1:, 1:]
data = data.astype(float)

#standardize the data using z scores (z = (x - mean) / std)
data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)


#step 2: compute pairwise distances(euclidean and manhattan)

#find number of samples for matrix size
n = data.shape[0]

#initialize empty matrix and collection matrix
euclid_matrix = np.empty((n, n))
euc_dist = []

#function to compute distance(euclidean), based off slides formula
def euclid_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

#fill in the matrix
for i in range(n):
    for j in range(n):
        euclid_matrix[i, j] = euclid_dist(data[i], data[j])

        if j < i:
            euc_dist.append(euclid_matrix[i, j])

#initialize empty matrix and collection matrix
manhattan_matrix = np.empty((n, n))
man_dist = []

#function to compute distance(manhattan), based off slides formula
def manhattan_dist(x, y):
    return np.sum(np.abs(x - y))

#fill in the matrix
for i in range(n):
    for j in range(n):
        manhattan_matrix[i, j] = manhattan_dist(data[i], data[j])

        if j < i:
            man_dist.append(manhattan_matrix[i, j])
            

print("Euclidean Distance Values:")
print("minimum:", round(np.min(euc_dist), 4))
print("maximum:", round(np.max(euc_dist), 4))
print("mean:", round(np.mean(euc_dist), 4))
print("median:", round(np.median(euc_dist), 4))
print()
print()
print("Manhattan Distance Values:")
print("minimum:", round(np.min(man_dist), 4))
print("maximum:", round(np.max(man_dist), 4))
print("mean:", round(np.mean(man_dist), 4))
print("median:", round(np.median(man_dist), 4))


