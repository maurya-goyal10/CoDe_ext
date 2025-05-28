import torch

from sklearn.cluster import DBSCAN, KMeans
import hdbscan
import numpy as np

def cluster(curr_samples_np,method="KMeans"):
    if method=="KMeans":
        # print(method)
        l = curr_samples_np.shape[0]
        n_clusters = l // 2 if l >= 4 else l
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(curr_samples_np)
        # print(f"The shape of the labels is {labels.shape} and the n_clusters is {n_clusters}")
        return n_clusters,labels
    
    elif method=="HDBSCAN":
        # print(method)
        clusterer = hdbscan.HDBSCAN()
        labels = clusterer.fit_predict(curr_samples_np)
        
        # print(labels)
        labels_max = max(labels)
        indices = np.where(labels == -1)[0]
        # print(labels_max)
        # print(indices)
        labels[indices] = np.arange(labels_max+1,labels_max+1+len(indices))
        # print(labels)
        return len(set(labels)),labels
        
    else:
        print("Not yet executed!")
        
        
        
        
# curr_samples_np
# print(curr_samples.shape,curr_samples_np.shape)

# distances = []
# for i, j in combinations(range(4), 2):
#     dist = np.linalg.norm(curr_samples_np[i] - curr_samples_np[j])
#     distances.append(dist)

# # print(np.array(distances))
# eps = 20*self.get_variance(t).item()
# print(f"The variance is {self.get_variance(t)} and the distances are {np.array(distances)}")
# db = DBSCAN(eps=eps, min_samples=2).fit(curr_samples_np)
# labels = db.labels_
# print(labels)