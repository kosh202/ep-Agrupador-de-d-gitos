import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Carrega o conjunto de dados MNIST
mnist = fetch_openml('mnist_784')
X, y = mnist.data.to_numpy(), mnist.target

# Seleciona algumas imagens aleatórias para plotar
indices = [0, 100, 200, 300, 400]  # Índices das imagens que você quer plotar

# Plota as imagens
fig, axs = plt.subplots(1, len(indices), figsize=(10, 3))

for i, idx in enumerate(indices):
    img = X[idx].reshape(28, 28)  # Redimensiona o vetor de características para uma matriz 28x28
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f'Label: {y[idx]}')

plt.tight_layout()
plt.show()
