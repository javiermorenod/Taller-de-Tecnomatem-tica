import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

torch.manual_seed(0)

# ----------------------------
# 1️⃣ Generar datos no lineales
# ----------------------------
X, y = make_circles(n_samples=500, noise=0.1, factor=0.5)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ----------------------------
# 2️⃣ Modelo tipo PINN (más rígido)
# ----------------------------
class PINN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 3️⃣ Modelo tipo KAN (más flexible)
# ----------------------------
class KAN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 4️⃣ Función de entrenamiento
# ----------------------------
def train(model, X, y, epochs=2000, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# ----------------------------
# 5️⃣ Función para graficar frontera
# ----------------------------
def plot_decision_boundary(model, X, y, title, subplot_position):
    plt.subplot(subplot_position)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        Z = model(grid)
        Z = torch.sigmoid(Z)
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z.numpy(), levels=50, cmap="bwr", alpha=0.5)
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="bwr", edgecolors='k')
    plt.title(title)

# ----------------------------
# 6️⃣ Entrenamiento
# ----------------------------
pinn = PINN_Model()
kan = KAN_Model()

pinn = train(pinn, X, y)
kan = train(kan, X, y)

# ----------------------------
# 7️⃣ Visualización comparativa
# ----------------------------
plt.figure(figsize=(14,6))

plot_decision_boundary(pinn, X, y, "Modelo tipo PINN (más rígido)", 121)
plot_decision_boundary(kan, X, y, "Modelo tipo KAN (más flexible)", 122)

plt.tight_layout()
plt.show()
