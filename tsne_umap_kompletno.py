"""
t-SNE i UMAP — kompletna analiza, poređenje sa PCA i LDA, izuzeci i ograničenja.

Instalacija:  pip install umap-learn

Struktura:
  I   Uvod — učitavanje podataka, setup
  II  t-SNE i UMAP — vizualizacija i analiza
       1. t-SNE na Wine datasetu (osnovno)
       2. UMAP na Wine datasetu (osnovno)
       3. Utjecaj perplexity (t-SNE) i n_neighbors (UMAP)
       4. PCA vs LDA vs t-SNE vs UMAP — direktno poređenje
       5. Gdje t-SNE i UMAP pobjeđuju PCA (nelinearne strukture)
       6. Izuzeci t-SNE:
          - Lažni klasteri iz uniformnih podataka
          - Globalna struktura nije očuvana
          - Nedeterminizam
          - Nema transform() za nove tačke
       7. Izuzeci UMAP:
          - Osjetljivost na n_neighbors
          - Lažne veze između klastera pri malim n_neighbors
          - Osjetljivost na min_dist
       8. t-SNE vs UMAP — direktna usporedba prednosti/mana
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

os.makedirs("slike", exist_ok=True)

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from utils import confidence_ellipse, inter_intra_ratio, y, X_scaled, class_names, colors_3class, colors_2class

try:
    from umap import UMAP
    UMAP_DOSTUPAN = True
except ImportError:
    UMAP_DOSTUPAN = False
    print(" UMAP nije instaliran. Pokreni: pip install umap-learn")

np.random.seed(42)  # fiksira slucajnost da uvek imam iste rezultate

# ══════════════════════════════════════════════════════════════════════════════
#  I  UVOD — Dataset i zajedničke funkcije
# ══════════════════════════════════════════════════════════════════════════════

# za pca i lda
# PCA i LDA za poređenje
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# t-SNE projekcija - standard za viz klastera
print("Računam t-SNE projekciju...")
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
# perplexity broj susjeda koje tacka gleda - 5 mnogo sitnih rastrkanih grupica, 100 sira slika

# UMAP projekcija
if UMAP_DOSTUPAN:
    print("Računam UMAP projekciju...")
    umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    # min_dist koliko gusto tacke mogu diti spakovane uz drugu na grafu
    X_umap = umap_model.fit_transform(X_scaled)


def upozorenje(ax, tekst):
    """Dodaje žuto upozorenje u donji lijevi ugao grafa."""
    ax.text(0.02, 0.02, tekst, transform=ax.transAxes, fontsize=9, color="darkred",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 1 — t-SNE na Wine datasetu
# ══════════════════════════════════════════════════════════════════════════════
def figura_1_tsne_wine():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE na Wine datasetu", fontsize=16, fontweight="bold")

    ax = axes[0]
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                   alpha=0.8, edgecolors="white", linewidth=0.5, s=60)
        confidence_ellipse(X_tsne[mask, 0], X_tsne[mask, 1], ax, n_std=2.0,
                           alpha=0.15, facecolor=color, edgecolor=color, linewidth=2)
    r = inter_intra_ratio(X_tsne, y)
    ax.set_title(f"t-SNE projekcija (perplexity=30)\ninter/intra={r:.2f}", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    upozorenje(ax, "⚠ Osi t-SNE nemaju fizičko značenje!\nRastojanje između klastera nije pouzdano.")
    # hoce samo da zadrzi relativni odnos a ne vrijednosti
    # vjestacki 2d prostor

    # Poredjenje separabilnosti pca, lda, tsne
    ax = axes[1]
    r_pca  = inter_intra_ratio(X_pca, y)
    r_lda  = inter_intra_ratio(X_lda, y)
    r_tsne = inter_intra_ratio(X_tsne, y)
    metode  = ["PCA", "LDA", "t-SNE"]
    ratios  = [r_pca, r_lda, r_tsne]
    boje    = ["#457b9d", "#e63946", "#f4a261"]
    bars = ax.bar(metode, ratios, color=boje, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Separabilnost klasa (inter/intra)\nPCA vs LDA vs t-SNE", fontsize=12)
    ax.set_ylabel("Omjer disperzije")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ratios) * 1.25)

    plt.tight_layout()
    plt.savefig("slike/t01_tsne_wine.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 2 — UMAP na Wine datasetu - najnapredniji algoritam do sad
# ══════════════════════════════════════════════════════════════════════════════
def figura_2_umap_wine():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("UMAP na Wine datasetu", fontsize=16, fontweight="bold")

    ax = axes[0]
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, label=name,
                   alpha=0.8, edgecolors="white", linewidth=0.5, s=60)
        confidence_ellipse(X_umap[mask, 0], X_umap[mask, 1], ax, n_std=2.0,
                           alpha=0.15, facecolor=color, edgecolor=color, linewidth=2)
    r = inter_intra_ratio(X_umap, y)
    ax.set_title(f"UMAP projekcija (n_neighbors=15, min_dist=0.1)\ninter/intra={r:.2f}", fontsize=12)
    # apstraktne dimenzije koje nastoje da ocuvaju topologiju
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Poredjenje separabilnosti — sada i UMAP
    ax = axes[1]
    r_pca  = inter_intra_ratio(X_pca, y)
    r_lda  = inter_intra_ratio(X_lda, y)
    r_tsne = inter_intra_ratio(X_tsne, y)
    r_umap = inter_intra_ratio(X_umap, y)
    metode = ["PCA", "LDA", "t-SNE", "UMAP"]
    ratios = [r_pca, r_lda, r_tsne, r_umap]
    boje   = ["#457b9d", "#e63946", "#f4a261", "#9b5de5"]
    bars = ax.bar(metode, ratios, color=boje, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Separabilnost klasa (inter/intra)\nPCA vs LDA vs t-SNE vs UMAP", fontsize=12)
    ax.set_ylabel("Omjer disperzije")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ratios) * 1.25)

    plt.tight_layout()
    plt.savefig("slike/t02_umap_wine.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 3 — Utjecaj hyperparametara: perplexity (t-SNE) i n_neighbors (UMAP)
# ══════════════════════════════════════════════════════════════════════════════
def figura_3_hyperparametri():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Osjetljivost na hyperparametre\nt-SNE (perplexity) vs UMAP (n_neighbors)",
                 fontsize=15, fontweight="bold")

    # Gornji red: t-SNE s različitim perplexity
    perplexities = [5, 15, 30, 50]
    for j, perp in enumerate(perplexities):
        X_t = TSNE(n_components=2, perplexity=perp, random_state=42,
                   max_iter=1000).fit_transform(X_scaled)
        ax = axes[0, j]
        for i, (name, color) in enumerate(zip(class_names, colors_3class)):
            mask = y == i
            ax.scatter(X_t[mask, 0], X_t[mask, 1], c=color, label=name,
                       alpha=0.8, edgecolors="white", linewidth=0.5, s=40)
        r = inter_intra_ratio(X_t, y)
        ax.set_title(f"t-SNE\nperplexity={perp}  |  r={r:.2f}", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if j == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Donji red: UMAP s različitim n_neighbors
    if UMAP_DOSTUPAN:
        n_neighbors_range = [3, 10, 30, 100]
        for j, nn in enumerate(n_neighbors_range):
            X_u = UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                       random_state=42).fit_transform(X_scaled)
            ax = axes[1, j]
            for i, (name, color) in enumerate(zip(class_names, colors_3class)):
                mask = y == i
                ax.scatter(X_u[mask, 0], X_u[mask, 1], c=color, label=name,
                           alpha=0.8, edgecolors="white", linewidth=0.5, s=40)
            r = inter_intra_ratio(X_u, y)
            ax.set_title(f"UMAP\nn_neighbors={nn}  |  r={r:.2f}", fontsize=10)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            if j == 0:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    else:
        for j in range(4):
            axes[1, j].text(0.5, 0.5, "UMAP nije instaliran",
                            ha="center", va="center", transform=axes[1, j].transAxes)

    plt.tight_layout()
    plt.savefig("slike/t03_hyperparametri.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 4 — PCA vs LDA vs t-SNE vs UMAP — direktno poredjenje
# ══════════════════════════════════════════════════════════════════════════════
def figura_4_poredenje():
    ncols = 4 if UMAP_DOSTUPAN else 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 6))
    fig.suptitle("PCA vs LDA vs t-SNE vs UMAP — isti podaci (Wine dataset)",
                 fontsize=16, fontweight="bold")

    datasets = [
        (X_pca,  "PCA\n(traži varijansu)",           "PC1",     "PC2"),
        (X_lda,  "LDA\n(traži separaciju klasa)",     "LD1",     "LD2"),
        (X_tsne, "t-SNE\n(lokalna susjedstva)",        "t-SNE 1", "t-SNE 2"),
    ]
    if UMAP_DOSTUPAN:
        datasets.append((X_umap, "UMAP\n(lokalna + globalna)", "UMAP 1", "UMAP 2"))

    for ax, (X_proj, title, xlabel, ylabel) in zip(axes, datasets):
        for i, (name, color) in enumerate(zip(class_names, colors_3class)):
            mask = y == i
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1], c=color, label=name,
                       alpha=0.8, edgecolors="white", linewidth=0.5, s=60)
            confidence_ellipse(X_proj[mask, 0], X_proj[mask, 1], ax, n_std=2.0,
                               alpha=0.15, facecolor=color, edgecolor=color, linewidth=2)
        r = inter_intra_ratio(X_proj, y)
        ax.set_title(f"{title}\ninter/intra={r:.2f}", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("slike/t04_poredenje.png", dpi=150, bbox_inches="tight")


def figura_5_nelinearne():
    datasets_nl = [
        ("Polumjeseci (moons)",  *make_moons(n_samples=300,   noise=0.1,  random_state=42)),
        ("Koncentrični krugovi", *make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)),
    ]

    # 5 kolona
    ncols = 5 if UMAP_DOSTUPAN else 4
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 4, 10))
    fig.suptitle("Linearno (PCA, LDA) vs Nelinearno (t-SNE, UMAP)\nČak i nadgledani LDA zakazuje na nelinearnim strukturama",
                 fontsize=15, fontweight="bold")

    for row, (naziv, X_d, y_d) in enumerate(datasets_nl):
        X_d_sc = StandardScaler().fit_transform(X_d)

        # 1. Originalni (2D)
        ax = axes[row, 0]
        for i, color in enumerate(colors_2class):
            mask = y_d == i
            ax.scatter(X_d[mask, 0], X_d[mask, 1], c=color, alpha=0.7, label=f"Kl {i}")
        ax.set_title("Original (2D)")

        # 2. PCA (1D)
        X_p = PCA(n_components=1).fit_transform(X_d_sc)
        axes[row, 1].scatter(X_p[:, 0], np.zeros_like(X_p[:, 0]), c=[colors_2class[i] for i in y_d], alpha=0.7)
        axes[row, 1].set_title(f"PCA (1D)\nr={inter_intra_ratio(X_p, y_d):.2f}")

        # 3. LDA (1D)
        # LDA na 2 klase moze imati samo 1 komponentu (C-1)
        X_l = LDA(n_components=1).fit_transform(X_d_sc, y_d)
        axes[row, 2].scatter(X_l[:, 0], np.zeros_like(X_l[:, 0]), c=[colors_2class[i] for i in y_d], alpha=0.7)
        axes[row, 2].set_title(f"LDA (1D)\nr={inter_intra_ratio(X_l, y_d):.2f}")

        # 4. t-SNE (1D)
        X_t = TSNE(n_components=1, random_state=42).fit_transform(X_d_sc)
        axes[row, 3].scatter(X_t[:, 0], np.zeros_like(X_t[:, 0]), c=[colors_2class[i] for i in y_d], alpha=0.7)
        axes[row, 3].set_title(f"t-SNE (1D)\nr={inter_intra_ratio(X_t, y_d):.2f}")

        # 5. UMAP (1D)
        if UMAP_DOSTUPAN:
            X_u = UMAP(n_components=1, random_state=42).fit_transform(X_d_sc)
            axes[row, 4].scatter(X_u[:, 0], np.zeros_like(X_u[:, 0]), c=[colors_2class[i] for i in y_d], alpha=0.7)
            axes[row, 4].set_title(f"UMAP (1D)\nr={inter_intra_ratio(X_u, y_d):.2f}")

    plt.tight_layout()
    plt.savefig("slike/t05_nelinearne_komplet.png", dpi=150)

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 6 — Izuzeci t-SNE
# ══════════════════════════════════════════════════════════════════════════════
def figura_6_izuzeci_tsne():
    fig, axes = plt.subplots(1, 1, figsize=(9, 6))
    fig.suptitle("Izuzeci i ograničenja t-SNE", fontsize=16, fontweight="bold")

    # Problem 2: globalna struktura nije ocuvana
    # uvecao crvene, suybio yeleni, i plavo je na istom rastojanju od crvenog i zelenog klastera ali na grafu nije tako
    ax = axes
    # raylicit broj i gustina tacaka
    X_blobs, y_blobs = make_blobs(n_samples=[50, 150, 300],
                                  centers=[[-10, 0], [0, 0], [10, 0]],
                                  cluster_std=[0.5, 2.0, 4.0], random_state=42)
    X_b_sc = StandardScaler().fit_transform(X_blobs)
    X_t_b  = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_b_sc)
    blob_labels = ["Mali klaster (50)", "Srednji (150)", "Veliki (300)"]
    for i, (color, label) in enumerate(zip(colors_3class, blob_labels)):
        mask = y_blobs == i
        ax.scatter(X_t_b[mask, 0], X_t_b[mask, 1], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, s=50, label=label)
    ax.set_title("Problem 2: Globalna struktura izgubljena\nOriginalno: mali—srednji—veliki su na pravoj liniji\nt-SNE mijenja redosljed i veličinu klastera\nRastojanje između klastera nije pouzdano",
                 fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.02, "Veličina klastera u t-SNE\nne odgovara stvarnoj veličini!",
            transform=ax.transAxes, fontsize=9, color="darkred",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig("slike/t06_izuzeci_tsne.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 7 — Izuzeci UMAP
# ══════════════════════════════════════════════════════════════════════════════
def figura_7_izuzeci_umap():
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Izuzeci i ograničenja UMAP", fontsize=16, fontweight="bold")

    # Problem 1: Mali n_neighbors — lazne veze i fragmentirani klasteri
    ax = axes[0]
    # vidi samo 2 najblize tacke
    X_u_mali = UMAP(n_components=2, n_neighbors=2, min_dist=0.1,
                    random_state=42).fit_transform(X_scaled)
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i
        ax.scatter(X_u_mali[mask, 0], X_u_mali[mask, 1], c=color, label=name,
                   alpha=0.8, edgecolors="white", linewidth=0.5, s=50)
        # fisher score mali jer je sve iscjeppkano
    r = inter_intra_ratio(X_u_mali, y)
    ax.set_title(f"Problem 1: Premali n_neighbors=2\nKlasteri se raspadaju na fragmente\ninter/intra={r:.2f}",
                 fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    upozorenje(ax, "Premali n_neighbors → lokalni šum\ndominira, klasteri se raspadaju!")

    # Problem 2: Utjecaj min_dist — kompaktnost klastera
    ax = axes[1]
    min_dists = [0.0, 0.1, 0.8] # razlicite gustine
    md_colors = ["#e63946", "#457b9d", "#2a9d8f"]
    md_labels = ["min_dist=0.0 (zbijeno)", "min_dist=0.1 (standard)", "min_dist=0.8 (razbacano)"]
    for md, color, label in zip(min_dists, md_colors, md_labels):
        X_u_md = UMAP(n_components=2, n_neighbors=15, min_dist=md,
                      random_state=42).fit_transform(X_scaled)
        mask = y == 0  # samo klasu 0 za preglednost
        ax.scatter(X_u_md[mask, 0], X_u_md[mask, 1], c=color, alpha=0.6,
                   edgecolors="white", linewidth=0.3, s=40, label=label)
    ax.set_title("Problem 2: Utjecaj min_dist\nIsta klasa (0), različit min_dist\n→ različita kompaktnost klastera",
                 fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    upozorenje(ax, "min_dist kontroliše 'gustoću' klastera\nNema 'ispravne' vrijednosti.")

    plt.tight_layout()
    plt.savefig("slike/t07_izuzeci_umap.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 8 — Projekcija novih tačaka (Out-of-sample mapping)
# ══════════════════════════════════════════════════════════════════════════════
def figura_8_transform():
    # 4 ako imamo UMAP, 3 ako nemamo
    ncols = 4 if UMAP_DOSTUPAN else 3
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 6))
    fig.suptitle("Projekcija novih tačaka: PCA ✓ | t-SNE ✗ | LDA ✓ | UMAP ✓",
                 fontsize=15, fontweight="bold")

    # Delimo podatke na Trening (140 uzoraka) - stari podaci i Test (ostatak - nove tacke)
    np.random.seed(42)
    idx = np.random.permutation(len(X_scaled))
    train_idx, test_idx = idx[:140], idx[140:]
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # funkcija za crtanje panela
    def nacrtaj_panel(ax, X_tr_proj, X_te_proj, y_tr, title, xlabel, ylabel, warning=None):
        # trening podaci
        for i, color in enumerate(colors_3class):
            mask = y_tr == i
            ax.scatter(X_tr_proj[mask, 0], X_tr_proj[mask, 1],
                       c=color, alpha=0.4, s=40, label=f"Train klasa {i}")

        # nove tacke crtamo kao crne zvezde
        ax.scatter(X_te_proj[:, 0], X_te_proj[:, 1],
                   c="black", marker="*", s=150, zorder=5, label="Nove tačke")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        if warning:
            upozorenje(ax, warning)

    # 1. PCA PANEL
    pca_tr = PCA(n_components=2).fit(X_train)
    nacrtaj_panel(axes[0],
                  pca_tr.transform(X_train),
                  pca_tr.transform(X_test),
                  y_train,
                  "PCA ✓\npca.fit(X_train) → transform(X_test)",
                  "PC1", "PC2")

    # 2. t-SNE PANEL (Nema transform!)
    # Moramo fitovati sve zajedno jer t-SNE ne ume da doda nove tacke naknadno
    X_all_tsne = TSNE(n_components=2, perplexity=30,
                      random_state=42).fit_transform(np.vstack([X_train, X_test]))
    nacrtaj_panel(axes[1],
                  X_all_tsne[:140],
                  X_all_tsne[140:],
                  y_train,
                  "t-SNE ✗\nMora ponovo trenirati na svim podacima\nTSNE().fit_transform(train + test)",
                  "t-SNE 1", "t-SNE 2",
                  "Nema transform()!\nNe možeš dodati novu tačku\nbez ponovnog treninga.")

    # 3. LDA PANEL
    lda_tr = LDA(n_components=2).fit(X_train, y_train)
    nacrtaj_panel(axes[2],
                  lda_tr.transform(X_train),
                  lda_tr.transform(X_test),
                  y_train,
                  "LDA ✓\nlda.fit(train) → transform(test)",
                  "LD1", "LD2")

    # 4. UMAP PANEL (Ako je dostupan)
    if UMAP_DOSTUPAN:
        umap_tr = UMAP(n_components=2, n_neighbors=15, random_state=42).fit(X_train)
        nacrtaj_panel(axes[3],
                      umap_tr.transform(X_train),
                      umap_tr.transform(X_test),
                      y_train,
                      "UMAP ✓\numap.fit(X_train) → transform(X_test)",
                      "UMAP 1", "UMAP 2")

    plt.tight_layout()
    plt.savefig("slike/t08_transform.png", dpi=150, bbox_inches="tight")






# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generisanje figura...\n")
    if not UMAP_DOSTUPAN:
        print("  Napomena: UMAP nije instaliran — neke figure bit će preskočene.")
        print("  Instaliraj sa: pip install umap-learn\n")

    figura_1_tsne_wine()
    print("  ✓ Figura 1: t-SNE na Wine (t01_tsne_wine.png)")

    figura_2_umap_wine()
    print("  ✓ Figura 2: UMAP na Wine (t02_umap_wine.png)")

    figura_3_hyperparametri()
    print("  ✓ Figura 3: Hyperparametri — perplexity i n_neighbors (t03_hyperparametri.png)")

    figura_4_poredenje()
    print("  ✓ Figura 4: PCA vs LDA vs t-SNE vs UMAP (t04_poredenje.png)")

    figura_5_nelinearne()
    print("  ✓ Figura 5: Nelinearne strukture (t05_nelinearne.png)")

    figura_6_izuzeci_tsne()
    print("  ✓ Figura 6: Izuzeci t-SNE (t06_izuzeci_tsne.png)")

    figura_7_izuzeci_umap()
    print("  ✓ Figura 7: Izuzeci UMAP (t07_izuzeci_umap.png)")

    figura_8_transform()
    print("  ✓ Figura 8: transform() — nove tačke (t08_transform.png)")

    print("\n" + "=" * 60)
    print("  Separabilnost klasa (Wine dataset)")
    print("=" * 60)
    print(f"  PCA   (2 komponente) : {inter_intra_ratio(X_pca, y):.4f}")
    print(f"  LDA   (2 komponente) : {inter_intra_ratio(X_lda, y):.4f}")
    print(f"  t-SNE (perp=30)      : {inter_intra_ratio(X_tsne, y):.4f}")
    if UMAP_DOSTUPAN:
        print(f"  UMAP  (nn=15)        : {inter_intra_ratio(X_umap, y):.4f}")
    print("=" * 60)
    print("\n  Ključne razlike t-SNE vs UMAP:")
    print("  t-SNE: sporo, nema transform(), samo vizualizacija")
    print("  UMAP:  brže, ima transform(), bolja globalna struktura")
    print("  Oboje: nelinearni, stohastični, hyperparametri bitni")
