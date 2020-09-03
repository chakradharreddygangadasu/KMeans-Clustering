# KMeans Clustering

"""The mall data set consists of independent variables 'CustomerID', 'Genre', 'Age', 'Annual Income (k$)',
'Spending Score (1-100)' so based on the available features we need to make clusters that would help the
organization for the development. Since there is no dependent variable its called unsupervised learning. """


##immporting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

##first we need to know how many clusters to be done by seeing the wcss(with in cluster sum of square) elbow method
##ploting wcss wrt number of cluster
##using the elbow method to find the optimal number of cluster

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10,random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
print(wcss)
plt.plot(range(1,11),wcss, color = 'blue')
plt.title('elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

##applyting k-means to the mall data(since we got the optimum number of clusters)
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init= 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

##visualizing the clusters
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans ==0,1], color = 'red', s = 100, label = 'cluster1')
plt.scatter(x[y_kmeans ==1,0], x[y_kmeans == 1, 1], color = 'blue', s = 100, label = 'cluster2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans== 2, 1], color = 'green', s = 100, label = 'cluster3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3,1], color = 'cyan', s = 100, label = 'cluster4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], color = 'magenta', s = 100, label = 'cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s =300, c = 'yellow', label = 'centroids')
plt.title('clusters of clints')
plt.xlabel('annual income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

