import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#load the data
data = np.loadtxt('CIS4020F25DataSet.csv', delimiter=',', dtype=str)

#remove the identifiers, the ID's and convert the values to floats
#source: https://www.w3schools.com/python/numpy/numpy_array_slicing.asp (slicing arrays)
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

manhattan_matrix = np.empty((n, n))
man_dist = []

#Keeping seperate for clarity, could condense

#fill in the matrix
for i in range(n):
    for j in range(n):
        euclid_matrix[i, j] = np.sqrt(np.sum((data[i] - data[j]) ** 2))

        if j < i:
            euc_dist.append(euclid_matrix[i, j])

#fill in the matrix
for i in range(n):
    for j in range(n):
        manhattan_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))

        if j < i:
            man_dist.append(manhattan_matrix[i, j])
            

#print results, round
print("Euclidean Distance Values:")
print("minimum:", round(np.min(euc_dist), 6))
print("maximum:", round(np.max(euc_dist), 6))
print("mean:", round(np.mean(euc_dist), 6))
print("median:", round(np.median(euc_dist), 6))
print()
print()
print("Manhattan Distance Values:")
print("minimum:", round(np.min(man_dist), 6))
print("maximum:", round(np.max(man_dist), 6))
print("mean:", round(np.mean(man_dist), 6))
print("median:", round(np.median(man_dist), 6))


#step 3: PCA (PC1, PC2, PC3)
#we can simplify this by using scikit learn

#k val for PC1, PC2, PC3
k = 3

#compute PCA with k components
pca = PCA(n_components=k)

#fit and transform the data for printing
pca_rough = pca.fit(data)
pca_result = pca.fit_transform(data)

#create a table of variance explaining the components(3), the variance, and percentage of variance
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
components = pca.components_


print()
print("PCA Component Table:")
print("Component\tVariance\t% of Variance")
for i in range(k):
    print(f"PC{i+1}\t\t\t{round(explained_variance[i], 6)}\t{round(explained_variance_ratio[i]*100, 6)}%")
print()