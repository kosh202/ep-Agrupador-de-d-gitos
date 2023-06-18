import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Digits Dataset
digits = load_digits()
data = digits.data
target = digits.target

# Reduzir a dimensionalidade dos dados para 2 usando PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Plotar o gráfico 2D antes do K-means
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], cmap='viridis')
plt.title('Data Plot - Digits Dataset')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Criar uma instância do algoritmo K-means
kmeans = KMeans(n_clusters=10)  # Defina o número de clusters desejado

# Ajustar o modelo aos dados
kmeans.fit(data_2d)

# Obter as etiquetas de cluster para cada amostra
labels = kmeans.labels_

# Plotar o gráfico 2D com os pontos de dados coloridos de acordo com as etiquetas de cluster
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar(label='Cluster')

# Obter as coordenadas dos centroides
centroids = kmeans.cluster_centers_

# Plotar os centroides em vermelho
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c='red', s=100)

plt.title('K-means Clustering - Digits Dataset')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
