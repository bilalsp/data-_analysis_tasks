import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OrdinalEncoder, 
                                   FunctionTransformer, 
                                   StandardScaler)
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA

plt.style.use('ggplot')


class ClusterAnalysis:

    def __init__(self) -> None:
        """Cluster-analysis on Customer Data"""
        self.output_path = 'outputs/visuals/'
        self.data_path = 'data/customers.csv'

    def run(self, *args, verbose=True) -> None:
        data_col = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 
                    'Spending Score (1-100)']
        data = np.genfromtxt(
            self.data_path, delimiter=',', dtype=object, skip_header=1)

        data = data[:, 1:]  # remove CustomerID

        pipe = Pipeline(steps=[
            ('GenderBinarizer', ColumnTransformer(
                [('oe', OrdinalEncoder(), [0])], remainder='passthrough')),
            ('ChangeDtype', FunctionTransformer(lambda X: X.astype(np.float32))),
            ('StandardScaler', ColumnTransformer(
                [('ss', StandardScaler(), [1,2,3])], remainder='passthrough')),

        ], verbose=False)

        transformed_data = pipe.fit_transform(data)
        
        self._visualize_optimal_K(
            X=transformed_data, filename='optimal_value_of_k')

        # Add KMeans clustering algorithm in pipeline
        k = 4
        model = KMeans(
            n_clusters=k, init='k-means++', max_iter=20, n_init=10, 
            random_state=0)
        pipe.steps.append(('kmeans', model))

        # Fit and predict 
        pipe.verbose = True
        labels = pipe.fit_predict(data)
        cluster_centers = pipe['kmeans'].cluster_centers_
      
        self._plot_clusters(
            transformed_data, cluster_centers, labels, 
            title='Clusters Visualization')

        # (unscaled) Features mean of each cluster
        cluster_means = []
        pipe.verbose = False
        for i in range(k):
            cluster_means.append(
                pipe[:2].transform(data)[labels==i, :].mean(axis=0))
        cluster_means = np.array(cluster_means)
        
        # Observation: (Based on `cluster_means`)
        # Cluster-0 (or Segment-0) comprises those customers who has high 
        # income and spend alot. 
        # Customers in Cluster-1 earn high but they spend a little and 
        # there average age is higher than the Cluster-0.
        # Cluster-2 comprises customers who has medium income and spend 
        # at the same level. They are old people.
        # Cluster-3 comprises customers who has little income but they 
        # spend a lot. They are young people.

        if verbose:
            print('Visuals have been saved to {} directory.'.format(self.output_path))

    def _visualize_optimal_K(self, X, filename=None):
        fig = plt.figure()
        visualizer = KElbowVisualizer(
            KMeans(random_state=0), k=(1,13), timing=False)
        visualizer.fit(X) #Fit the data to the visualizer
        visualizer.show()  
        if filename:
            os.makedirs(self.output_path, exist_ok=True)
            fig.savefig(self.output_path+filename, dpi=fig.dpi) 


    def _plot_clusters(self, transformed_X, centroids, labels, title):
        # Apply PCA and fit the features
        pca_2d = PCA(n_components=2).fit_transform(transformed_X)
        
        fig, ax = plt.subplots(figsize=(10, 6))   
        plt.scatter(
            pca_2d[:, 0], 
            pca_2d[:, 1],
            c=labels, 
            cmap=plt.cm.get_cmap("Spectral_r", 5),
            alpha=0.5)
        ax.set_xlabel('PCA Component-1', fontsize=8, labelpad=10)      
        ax.set_ylabel('PCA Component-2', fontsize=8, labelpad=10)  

        fig.suptitle(title, fontweight ="bold", fontsize=16)
        plt.colorbar(ticks=list(range(len(centroids))))
        plt.show()
