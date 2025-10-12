import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

#load the data
data = np.loadtxt("CIS4020F25DataSet.csv", delimiter=",", dtype=str)

#remove the identifiers, the ID's and convert the values to floats
#source: https://www.w3schools.com/python/numpy/numpy_array_slicing.asp (slicing arrays)
data = data[1:, 1:]
data = data.astype(float)

#step 1:standardize the data using z scores (z = (x - mean) / std)
data = (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)

#step 2: compute euclidean and manhattan distances

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
print("Euclidean Distance Summary:")
print("minimum:", round(np.min(euc_dist), 3))
print("maximum:", round(np.max(euc_dist), 3))
print("mean:", round(np.mean(euc_dist), 3))
print("median:", round(np.median(euc_dist), 3))
print()
print("Manhattan Distance Summary:")
print("minimum:", round(np.min(man_dist), 3))
print("maximum:", round(np.max(man_dist), 3))
print("mean:", round(np.mean(man_dist), 3))
print("median:", round(np.median(man_dist), 3))
print()

#step 3: PCA (PC1, PC2, PC3)
#we can simplify this by using scikit learn

#k val for PC1, PC2, PC3
k = 3

#compute PCA with k components
pca = PCA(n_components=k)

#fit and transform the data for printing
pca_result = pca.fit_transform(data)

#create a table of variance explaining the components(3), the variance, and percentage of variance
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

#print the table to 3 decimal places
print("PCA Table of Variance:")
print("PC#        Variance        % of Variance")
print("PC1       ", round(explained_variance[0], 3), "         ", round(explained_variance_ratio[0] * 100, 3))
print("PC2       ", round(explained_variance[1], 3), "         ", round(explained_variance_ratio[1] * 100, 3))
print("PC3       ", round(explained_variance[2], 3), "         ", round(explained_variance_ratio[2] * 100, 3))
print()


#scatter plot of the first 3 components
#source: https://www.geeksforgeeks.org/python/3d-scatter-plotting-in-python-using-matplotlib/

#seperating our components by column
pc1 = pca_result[:, 0]
pc2 = pca_result[:, 1]
pc3 = pca_result[:, 2]

#form the plot, choosing green x's(because I like the colour green)
scatterPCA = plot.figure()
ax = scatterPCA.add_subplot(111, projection="3d")
ax.scatter(pc1, pc2, pc3, color="green", marker="x")

#labels for the plot
ax.set_title("PCA Scatter Plot")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plot.show()


#step 4: MDS 3D
#we can't use scikit learn, due to the MDS implementation only supporting metric/non-metric MDS
#compute Classical MDS (from formula in slides)
    
#square the euclidean distance matrix
d_square = euclid_matrix ** 2

#Calculate C matrix (I - 1/n * J)
n = d_square.shape[0]
i = np.eye(n)
j = np.ones((n, n))
c = i - (1/n) * j

#Calculate B matrix (-1/2 * C * D^2 * C)
b = -0.5 * c @ d_square @ c

#Eigen decomposition of B
#source: https://numpy.org/doc/2.3/reference/generated/numpy.linalg.eigh.html because i spent a horrible time learning the difference between eig and eigh functions(thank you numpy)
eigenvalues, eigenvectors = np.linalg.eigh(b)

#Sort eigenvalues and eigenvectors in descending order for top 3
#we can use argsort to find our indices, then flip and slice for our top 3 eigenvalues and eigenvectors
sort_values = np.argsort(eigenvalues)
flip_values = np.flip(sort_values)

#take the top 3
top_values = flip_values[:k]

#get our corresponding eigenvalues and eigenvectors
top3_eigenvalues = eigenvalues[top_values]
top3_eigenvectors = eigenvectors[:, top_values]

#Compute MDS result (X = E_3 * sqrt(^_3))
mds_result = top3_eigenvectors @ np.diag(np.sqrt(top3_eigenvalues))

#scatter plot of the top 3 eigen pairs

#seperating our components
mds1 = mds_result[:, 0]
mds2 = mds_result[:, 1]
mds3 = mds_result[:, 2]

#form the plot
scatterMDS = plot.figure()
ax = scatterMDS.add_subplot(111, projection="3d")
ax.scatter(mds1, mds2, mds3, color="green", marker="x")


#labels for the plot
ax.set_title("Classic MDS Scatter Plot")
ax.set_xlabel("MDS1")
ax.set_ylabel("MDS2")
ax.set_zlabel("MDS3")

plot.show()