import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

np.random.seed(42)

#  I  UVOD — Dataset i zajedničke funkcije
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
class_names = ["Barolo", "Grignolino", "Barbera"]

colors_3class = ["#e63946", "#457b9d", "#2a9d8f"]   # za wine
colors_2class = ["#e63946", "#457b9d"]
colors_5class = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#9b5de5"]

print(f"Dataset: Wine  |  uzoraka: {X.shape[0]}  |  featura: {X.shape[1]}  |  klasa: {len(class_names)}")
print(f"Klase: {class_names}\n")

# standardizacija - prosjek 0 i standardna devijacija 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2.0 je 95%
def confidence_ellipse(x, y_pts, ax, n_std=2.0, facecolor="none", **kwargs):
    """95% confidence elipsa oko klastera tačaka."""
    cov = np.cov(x, y_pts)  # kovarijanca - kako se x i y krecu zajedno
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])    # korelacija koliko su tacno povezane(od -1 do 1)
    # izgled elipse
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std    # standardna devijacija prosirena da  "pokupi" 95% tacaka
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y_pts)
    # namjesti elipsu da bude kako treba oblikom , velicinom i mjestom
    t = (transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
         .translate(mean_x, mean_y))
    ellipse.set_transform(t + ax.transData)
    return ax.add_patch(ellipse)


def inter_intra_ratio(X_proj, y):
    # J = rasipanje IZMEĐU klasa / rasipanje UNUTAR klasa
    if X_proj.ndim == 1:    # ako ije 1D pretvori u kolonu
        X_proj = X_proj.reshape(-1, 1)
    overall_mean = X_proj.mean(axis=0)
    classes = np.unique(y)
    # izmedju klasa - sto je veci vise su udaljene
    # kvadriramo da se + i - ne poniste
    # np.sum(y == c) broj uzoraka vece klase vise doprinose
    inter = sum(np.sum(y == c) * np.sum((X_proj[y == c].mean(axis=0) - overall_mean) ** 2)
                for c in classes)
    # unutar klasa
    # sto je manji broj ima vise tacaka oko centra
    intra = sum(np.sum((X_proj[y == c] - X_proj[y == c].mean(axis=0)) ** 2)
                for c in classes)

    # velik - odvojene i kompaktne klase
    # mali br - preklapaju se i raybacane su
    # 0 kada su npr koncentricni krugovi u pitanju
    return inter / intra if intra > 0 else 0
