"""
PCA i LDA — kompletno poređenje na realnim i vještački konstruisanim podacima.

Struktura:
  I   Uvod — učitavanje podataka, setup
  II  Klasične metode redukcije — PCA i LDA
       1. PCA na Wine datasetu
       2. LDA na Wine datasetu (separabilnost + KNN tačnost)
       3. Podaci koji ruše PCA (moons, signal u maloj varijanci)
       4. Podaci koji ruše LDA (krugovi, XOR)
       5. PCA fail + LDA win — signal u smjeru male varijanse
       6. Loadings heatmap (interpretabilnost)
       7. PCA → LDA pipeline (Fisherfaces pristup)
       8. Curse of dimensionality
       9. Osjetljivost na outliere
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("slike", exist_ok=True)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from utils import confidence_ellipse, inter_intra_ratio, wine, X, y, X_scaled, feature_names, class_names, colors_3class, colors_2class

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
#  I  UVOD — Dataset i zajedničke funkcije
# ══════════════════════════════════════════════════════════════════════════════

pca = PCA(n_components=2)   # samo 2 nova stupca jer je najlakse nacrtati
X_pca = pca.fit_transform(X_scaled)
pca_full = PCA().fit(X_scaled)  # koliko koja komponenta nosi informacija

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)  # y sorte vina da bi znao da ih razdvoji
lda_full = LDA().fit(X_scaled, y)   # za postotak uspjesnosti


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURA 1 — PCA na Wine datasetu
# ═════════════════════════════════════════════════════════════════════════════

# 178 boca - 3 regije - 13 stvari
# sa 13 na 2 dimenzije

def figura_1_pca_wine():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PCA – Principal Component Analysis (Wine dataset)",
                 fontsize=16, fontweight="bold")

    # sirovi podaci uzela sam prva 2 svojstva
    ax = axes[0]
    # prodji kroz klase
    # zip / spaja dvije liste  parove
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i   # maska True i False, True samo za uzorke koje pripadaju klasi
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=color, label=name,
                   alpha=0.7, edgecolors="white", linewidth=0.5)    # % alkohola i kiseline

    # strelice PC1 i PC2 na originalnom scatter-u
    mean0 = X_scaled[:, 0].mean()
    mean1 = X_scaled[:, 1].mean()
    scale = 2.0  # duzina strelice
    for j, (color, label) in enumerate(zip(["#e63946", "#2a9d8f"], ["PC1", "PC2"])):
        dx = pca.components_[j, 0] * scale * np.sqrt(pca.explained_variance_[j])
        dy = pca.components_[j, 1] * scale * np.sqrt(pca.explained_variance_[j])
        ax.annotate("", xy=(mean0 + dx, mean1 + dy), xytext=(mean0, mean1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        ax.text(mean0 + dx * 1.1, mean1 + dy * 1.1, label,
                color=color, fontsize=11, fontweight="bold")

    ax.set_title(f"Originalni podaci + PC smjerovi\n({feature_names[0]} vs {feature_names[1]})", fontsize=12)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # PCA smanjuje u 2D
    ax = axes[1]
    var1, var2 = pca.explained_variance_ratio_  # koliko % informacija nosi koja osa
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                   alpha=0.7, edgecolors="white", linewidth=0.5)
        confidence_ellipse(X_pca[mask, 0], X_pca[mask, 1], ax, n_std=2.0,
                           alpha=0.15, facecolor=color, edgecolor=color, linewidth=2)   # elipsa koja obuhvata 95%
    ax.set_title(f"PCA projekcija\nPC1={var1:.1%} | PC2={var2:.1%} | ukupno={var1+var2:.1%}",
                 fontsize=12)
    ax.set_xlabel(f"PC1 ({var1:.1%} varijance)")
    ax.set_ylabel(f"PC2 ({var2:.1%} varijance)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # dijagram odluke
    ax = axes[2]
    ev = pca_full.explained_variance_ratio_ # koliko % varijance nosi svaka komponenta ponaosob - PC13
    cumev = np.cumsum(ev)
    components = np.arange(1, len(ev) + 1)
    bars = ax.bar(components, ev * 100, color="#457b9d", alpha=0.8,
                  label="Individualna varijanca")
    ax.plot(components, cumev * 100, "o-", color="#e63946",
            linewidth=2, markersize=8, label="Kumulativna varijanca")
    ax.axhline(95, color="gray", linestyle="--", alpha=0.7, label="95% prag")
    for bar, val in zip(bars, ev):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Scree plot\n(objašnjena varijanca po komponentama)", fontsize=12)
    ax.set_xlabel("Broj komponenti (PC)")
    ax.set_ylabel("Objašnjena varijanca (%)")
    ax.set_xticks(components)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("slike/01_pca_wine.png", dpi=150, bbox_inches="tight")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURA 2 — LDA na Wine datasetu
# ═════════════════════════════════════════════════════════════════════════════
def figura_2_lda_wine():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("LDA – Linear Discriminant Analysis (Wine dataset)",
                 fontsize=16, fontweight="bold")

    # 2 komponente alkohol i kiselina
    ax = axes[0]
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=color, label=name,
                   alpha=0.7, edgecolors="white", linewidth=0.5)
    ax.set_title(f"Originalni podaci\n({feature_names[0]} vs {feature_names[1]})", fontsize=12)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # LDA = br klasa - 1
    ax = axes[1]
    for i, (name, color) in enumerate(zip(class_names, colors_3class)):
        mask = y == i
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=name,
                   alpha=0.7, edgecolors="white", linewidth=0.5)
        confidence_ellipse(X_lda[mask, 0], X_lda[mask, 1], ax, n_std=2.0,
                           alpha=0.15, facecolor=color, edgecolor=color, linewidth=2)
    ld_var = lda_full.explained_variance_ratio_
    ax.set_title(f"LDA projekcija\nLD1={ld_var[0]:.1%} | LD2={ld_var[1]:.1%} | ukupno={sum(ld_var):.1%}",
                 fontsize=12)
    ax.set_xlabel(f"LD1 ({ld_var[0]:.1%} separacije)")
    ax.set_ylabel(f"LD2 ({ld_var[1]:.1%} separacije)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("slike/02_lda_wine.png", dpi=150, bbox_inches="tight")

    # ===============================================================================
    # Separabilnost + klasifikacijska tacnosti
    # ===============================================================================
    # koliko su klase odvojene
    ratio_orig = inter_intra_ratio(X_scaled[:, :2], y)
    ratio_pca  = inter_intra_ratio(X_pca, y)
    ratio_lda  = inter_intra_ratio(X_lda, y)

    return ratio_orig, ratio_pca, ratio_lda


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 3 — Podaci koji ruše PCA (Redukcija na 1D)
# ══════════════════════════════════════════════════════════════════════════════
def figura_3_pca_failures():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Podaci koji ruše PCA (Smanjenje sa 2D na 1D)", fontsize=16, fontweight="bold")

    # --- Slučaj A: Polumjeseci (nelinearna struktura) ---
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_moons_scaled = StandardScaler().fit_transform(X_moons)

    # Originalni podaci (ostaju 2D radi konteksta)
    ax = axes[0, 0]
    for i, color in enumerate(colors_2class):
        mask = y_moons == i
        ax.scatter(X_moons_scaled[mask, 0], X_moons_scaled[mask, 1], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title("Slučaj A: Polumjeseci\n(Originalni 2D podaci)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PCA redukcija na 1D
    pca_moons = PCA(n_components=1)
    X_moons_pca = pca_moons.fit_transform(X_moons_scaled)

    ax = axes[0, 1]
    for i, color in enumerate(colors_2class):
        mask = y_moons == i
        # Crtamo na 1D liniji koristeći nule za Y osu
        ax.scatter(X_moons_pca[mask, 0], np.zeros_like(X_moons_pca[mask, 0]), c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")

    r = inter_intra_ratio(X_moons_pca, y_moons)
    ax.set_title(f"PCA (1D projekcija)\ninter/intra={r:.2f} — Klase su isprepletene!", fontsize=12)
    ax.set_xlabel("PC1 (Glavna komponenta)")
    ax.set_yticks([]) # Isključujemo Y osu jer je besmislena u 1D
    ax.legend()
    ax.grid(True, alpha=0.3)


    # --- Slučaj B: Varijansa šuma > Varijansa signala ---
    np.random.seed(42)
    n_b = 150
    X_b0 = np.column_stack([np.random.randn(n_b) * 0.3 - 1.5, np.random.randn(n_b) * 4])
    X_b1 = np.column_stack([np.random.randn(n_b) * 0.3 + 1.5, np.random.randn(n_b) * 4])
    X_noise = np.vstack([X_b0, X_b1])
    y_noise = np.array([0] * n_b + [1] * n_b)

    # Originalni podaci (2D)
    ax = axes[1, 0]
    for i, color in enumerate(colors_2class):
        mask = y_noise == i
        ax.scatter(X_noise[mask, 0], X_noise[mask, 1], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title("Slučaj B: Varijansa šuma (Y) > Varijansa signala (X)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PCA redukcija na 1D
    pca_noise = PCA(n_components=1)
    X_noise_pca = pca_noise.fit_transform(X_noise)
    v1 = pca_noise.explained_variance_ratio_[0]

    ax = axes[1, 1]
    for i, color in enumerate(colors_2class):
        mask = y_noise == i
        ax.scatter(X_noise_pca[mask, 0], np.zeros_like(X_noise_pca[mask, 0]), c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")

    r = inter_intra_ratio(X_noise_pca, y_noise)
    ax.set_title(f"PCA (1D) — Zadržan samo ŠUM!\nPC1={v1:.1%} varijance | inter/intra={r:.2f}", fontsize=12)
    ax.set_xlabel("PC1 (Pravac najveće varijanse = ŠUM)")
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("slike/03_pca_failures.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 4 — Podaci koji ruse LDA
# ══════════════════════════════════════════════════════════════════════════════
def figura_4_lda_failures():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Podaci koji ruše LDA", fontsize=16, fontweight="bold")

    # Koncentrični krugovi - problem centar im je isti kako da ovuce liniju da ih razdvoji
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    X_circles_scaled = StandardScaler().fit_transform(X_circles)

    # osnovni podaci
    ax = axes[0, 0]
    for i, color in enumerate(colors_2class):
        mask = y_circles == i
        ax.scatter(X_circles_scaled[mask, 0], X_circles_scaled[mask, 1], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title("Slučaj A: Koncentrični krugovi\n"
                 "(sredine klasa su identične!)", fontsize=12)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    lda_circles = LDA(n_components=1)
    X_circles_lda = lda_circles.fit_transform(X_circles_scaled, y_circles)
    np.random.seed(0)
    jitter_c = np.random.randn(len(y_circles)) * 0.05   # malo pomjeranje po y osi kako bi se tacke vidjele, tj ne bi bile na istoj horizontalnoj liniji

    ax = axes[0, 1]
    for i, color in enumerate(colors_2class):
        mask = y_circles == i
        ax.scatter(X_circles_lda[mask, 0], jitter_c[mask], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    r = inter_intra_ratio(X_circles_lda, y_circles)
    ax.set_title(f"LDA projekcija (1D)\ninter/intra={r:.2f} — klase se PREKLAPAJU!", fontsize=12)
    ax.set_xlabel("LD1")
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Slučaj B: XOR problem - isti centar za obje klase
    # LDa radi samo kada su centri klasa na raylicitim mjestima
    np.random.seed(42)
    n_x = 100
    # podaci
    X_xor = np.vstack([
        np.random.randn(n_x, 2) * 0.5 + [1, 1],
        np.random.randn(n_x, 2) * 0.5 + [-1, -1],
        np.random.randn(n_x, 2) * 0.5 + [1, -1],
        np.random.randn(n_x, 2) * 0.5 + [-1, 1],
    ])
    y_xor = np.array([0] * n_x + [0] * n_x + [1] * n_x + [1] * n_x)
    X_xor_scaled = StandardScaler().fit_transform(X_xor)

    ax = axes[1, 0]
    for i, color in enumerate(colors_2class):
        mask = y_xor == i
        ax.scatter(X_xor_scaled[mask, 0], X_xor_scaled[mask, 1], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title("Slučaj B: XOR problem\n"
                 "(obje klase imaju istu sredinu)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    lda_xor = LDA(n_components=1)
    X_xor_lda = lda_xor.fit_transform(X_xor_scaled, y_xor)
    np.random.seed(1)
    jitter_x = np.random.randn(len(y_xor)) * 0.05

    ax = axes[1, 1]
    for i, color in enumerate(colors_2class):
        mask = y_xor == i
        ax.scatter(X_xor_lda[mask, 0], jitter_x[mask], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    r = inter_intra_ratio(X_xor_lda, y_xor)
    ax.set_title(f"LDA projekcija (1D)\ninter/intra={r:.2f} — klase se PREKLAPAJU!", fontsize=12)
    ax.set_xlabel("LD1")
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("slike/04_lda_failures.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 5 — PCA fail + LDA win (artificial dataset)
#  Signal u smjeru male varijanse — PCA ga izgubi, LDA ne
#  jabuke na stolu primjer
# ══════════════════════════════════════════════════════════════════════════════
def figura_5_pca_fail_lda_win():
    """
    Konstruisemo podatke gdje:
      - Klase su odvojene u pravcu MALE varijanse (x osa: ±1, std 0.3)
      - Zajednička varijansa u pravcu VELIKE varijanse (y osa: std 3)

    PCA ce izabrati y osu kao PC1 (jer ima najveću varijansu) —
    ali tu osu klase dijele! Signal ostaje u PC2 koji bi se mogao baciti.

    LDA odmah identifikuje x osu kao pravac razdvajanja klasa.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PCA fail, LDA win — signal u smjeru male varijanse",
                 fontsize=16, fontweight="bold")

    np.random.seed(42)
    n = 200 # br tacaka u klasi
    X_s0 = np.column_stack([np.random.randn(n) * 0.3 - 1.0, np.random.randn(n) * 3.0])  # tacke oko -1, razbacane
    X_s1 = np.column_stack([np.random.randn(n) * 0.3 + 1.0, np.random.randn(n) * 3.0])
    X_sig = np.vstack([X_s0, X_s1])
    y_sig = np.array([0] * n + [1] * n) # prvih 200 su klasa 0 a drugih 200 su klasa 1

    # originalni podaci
    ax = axes[0]
    for i, color in enumerate(colors_2class):
        mask = y_sig == i
        ax.scatter(X_sig[mask, 0], X_sig[mask, 1], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title("Originalni podaci\n"
                 "signal u x osi (±1, std=0.3)\n"
                 "šum u y osi (std=3)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (signal)")
    ax.set_ylabel("y (šum)")

    # PCA — projiciramo samo na PC1 (1D) da pokažemo da bira šum
    pca_sig = PCA(n_components=2).fit(X_sig)
    X_sig_pca1 = pca_sig.transform(X_sig)[:, 0:1]   # samo PC1
    v1, v2 = pca_sig.explained_variance_ratio_
    r_pca = inter_intra_ratio(X_sig_pca1, y_sig)
    np.random.seed(3)
    jitter_pca = np.random.randn(len(y_sig)) * 0.08

    ax = axes[1]
    for i, color in enumerate(colors_2class):
        mask = y_sig == i
        ax.scatter(X_sig_pca1[mask, 0], jitter_pca[mask], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title(f"PCA — pogrešno!\n"
                 f"PC1={v1:.1%} varijance (= šum u y osi)\n"
                 f"inter/intra={r_pca:.3f} — klase PREKLOPLJENE", fontsize=11)
    ax.set_xlabel(f"PC1 ({v1:.1%} varijance)")
    ax.set_yticks([])
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # LDA — 1D projekcija koja hvata pravi signal
    lda_sig = LDA(n_components=1).fit(X_sig, y_sig)
    X_sig_lda = lda_sig.transform(X_sig)
    r_lda = inter_intra_ratio(X_sig_lda, y_sig)
    np.random.seed(7)
    jitter_lda = np.random.randn(len(y_sig)) * 0.08

    ax = axes[2]
    for i, color in enumerate(colors_2class):
        mask = y_sig == i
        ax.scatter(X_sig_lda[mask, 0], jitter_lda[mask], c=color, alpha=0.7,
                   edgecolors="white", linewidth=0.5, label=f"Klasa {i}")
    ax.set_title(f"LDA — ispravno!\n"
                 f"LD1 = signal u x osi\n"
                 f"inter/intra={r_lda:.3f}  ({r_lda/r_pca:.0f}× bolje od PCA)", fontsize=11)
    ax.set_xlabel("LD1")
    ax.set_yticks([])
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("slike/05_pca_fail_lda_win.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 6 — Loadings heatmap (interpretabilnost)
# ══════════════════════════════════════════════════════════════════════════════
def figura_6_loadings_heatmap():
    """
    Loadings pokazuju koliko svaki originalni feature doprinosi svakoj komponenti.
    Visoka apsolutna vrijednost = feature je bitan za tu komponentu.

    Ovo je ključno za INTERPETABILNOST — šta znači "PC1"?
    Ako je loadings za alkohol visok, PC1 mjeri nešto povezano sa alkoholom.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Loadings — koji feature-i najviše doprinose komponentama",
                 fontsize=16, fontweight="bold")

    # PCA loadings = eigenvektori pomnoženi sa √eigenvalue

    # prva 2 reda i 13 kolona
    # .T - transponira jer je lakse za citati
    # eigenvalues koliko varijanse nosi ta osa, veci eigenvalue znaci da je osa vaznija
    # mnozenje sa korjenom kako bi vektori bili razlicitih velicina,
    # eigenvektor strelica duzine 1, pokazuje smjer ali ne govori nista o vaznostu
    pca_loadings = pca_full.components_[:2].T * np.sqrt(pca_full.explained_variance_[:2])

    # LDA "loadings" = coef_ matrica
    lda_loadings = lda_full.scalings_[:, :2]    # svi feature

    # Panel 1: PCA
    ax = axes[0]
    im = ax.imshow(pca_loadings, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(pca_loadings).max(), vmax=np.abs(pca_loadings).max())   # simetricna skala boja
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PC1", "PC2"])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_title("PCA loadings", fontsize=12)

    # Vrijednosti u celije
    for i in range(len(feature_names)): # redovi - feature
        for j in range(2):  # kolone PC1 i PC2
            val = pca_loadings[i, j]    # pozicija
            # ako je > 50%max tamno crven ili plav kvadrat sa bijelim tekstom
            color = "white" if abs(val) > 0.5 * np.abs(pca_loadings).max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    plt.colorbar(im, ax=ax)

    # Panel 2: LDA
    ax = axes[1]
    im = ax.imshow(lda_loadings, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(lda_loadings).max(), vmax=np.abs(lda_loadings).max())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LD1", "LD2"])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_title("LDA scalings (slično loadings)", fontsize=12)

    for i in range(len(feature_names)):
        for j in range(2):
            val = lda_loadings[i, j]
            color = "white" if abs(val) > 0.5 * np.abs(lda_loadings).max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("slike/06_loadings_heatmap.png", dpi=150, bbox_inches="tight")

    def _pretty_heatmap(data, col_labels, title, filename):
        n_rows = data.shape[0]
        fig, ax = plt.subplots(figsize=(6, 8))
        fig.patch.set_facecolor("#f8f9fa")
        ax.set_facecolor("#f8f9fa")

        vmax = np.abs(data).max()
        im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

        # xtick oznake — veće i bold
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=13, fontweight="bold")
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", length=0, pad=6)

        # ytick oznake
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.tick_params(axis="y", length=0, pad=4)

        # tanke bijele linije između ćelija
        for x in np.arange(-0.5, len(col_labels), 1):
            ax.axvline(x, color="white", linewidth=2)
        for y in np.arange(-0.5, n_rows, 1):
            ax.axhline(y, color="white", linewidth=1)

        # vrijednosti unutar ćelija
        for i in range(n_rows):
            for j in range(len(col_labels)):
                val = data[i, j]
                txt_color = "white" if abs(val) > 0.55 * vmax else "#222222"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=txt_color)

        # colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(0)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=18, color="#1a1a2e")
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    _pretty_heatmap(pca_loadings, ["PC1", "PC2"], "PCA loadings", "slike/06a_pca_loadings.png")
    _pretty_heatmap(lda_loadings, ["LD1", "LD2"], "LDA scalings", "slike/06b_lda_scalings.png")



# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 7 — Curse of dimensionality
# ══════════════════════════════════════════════════════════════════════════════
def figura_7_curse_of_dim():
    """
    Pravimo dataset sa 100 feature-a od kojih je samo 5 informativno.
    Ostalih 95 je čist šum koji zbunjuje klasifikator.
    Mjeri se tačnost k-NN klasifikatora:
      - na svih 100 feature-a
      - na top 5 PCA komponenti
      - na top 2 LDA komponente
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Curse of dimensionality — kako redukcija pomaže k-NN klasifikatoru",
                 fontsize=15, fontweight="bold")

    # pravi dataset od 500 tacaka sa 100 featura
    X_high, y_high = make_classification(
        n_samples=500, n_features=100, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=2.0, random_state=42)
    X_high_scaled = StandardScaler().fit_transform(X_high)

    # k-NN tacnost sa razlicitim brojem dimenzija
    n_components_range = [1, 2, 3, 5, 10, 20, 50, 100]
    pca_scores = [] # lista tacnosti
    for n in n_components_range:
        if n <= min(X_high_scaled.shape):
            X_p = PCA(n_components=n).fit_transform(X_high_scaled)  # smanji podatke na n dimenzija
            score = cross_val_score(KNeighborsClassifier(n_neighbors=5),
                                    X_p, y_high, cv=5).mean()  # k-NN tacnost na reduciranim podacima
            pca_scores.append(score)
        else:
            pca_scores.append(None)

    # LDA ima samo C-1=2 komponente, ali ih koristim kao fiksnu tacku
    X_l = LDA(n_components=2).fit_transform(X_high_scaled, y_high)
    lda_score = cross_val_score(KNeighborsClassifier(n_neighbors=5),
                                X_l, y_high, cv=5).mean()

    # Baseline: svih 100 feature-a radimo knn
    baseline_score = cross_val_score(KNeighborsClassifier(n_neighbors=5),
                                     X_high_scaled, y_high, cv=5).mean()

    # Panel 1: tacnost vs broj PCA komponenti
    ax = axes[0]
    ax.plot(n_components_range, [s * 100 if s else None for s in pca_scores],
            "o-", color="#457b9d", linewidth=2, markersize=8, label="PCA + k-NN")
    ax.axhline(baseline_score * 100, color="gray", linestyle="--",
               label=f"Svih 100 feature-a ({baseline_score:.1%})")
    ax.axhline(lda_score * 100, color="#e63946", linestyle="--",
               label=f"LDA (2 komp., {lda_score:.1%})")
    ax.set_xscale("log")
    ax.set_xlabel("Broj komponenti")
    ax.set_ylabel("Tačnost k-NN (%)")
    ax.set_title("Tačnost vs broj dimenzija\n"
                 "(100 feature-a, samo 5 informativno)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: vizualizacija sa 2 PCA i LDA komponente
    ax = axes[1]
    methods_compare = ["Original\n(100D → k-NN)", "PCA\n(5 komp.)", "LDA\n(2 komp.)"]
    X_p5 = PCA(n_components=5).fit_transform(X_high_scaled)
    scores_compare = [
        baseline_score,
        cross_val_score(KNeighborsClassifier(n_neighbors=5), X_p5, y_high, cv=5).mean(),
        lda_score,
    ]
    bar_colors = ["#a8dadc", "#457b9d", "#e63946"]
    bars = ax.bar(methods_compare, [s * 100 for s in scores_compare],
                  color=bar_colors, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, scores_compare):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tačnost k-NN (cross-validation)")
    ax.set_title("Poređenje tačnosti", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("slike/07_curse_of_dim.png", dpi=150, bbox_inches="tight")



# ══════════════════════════════════════════════════════════════════════════════
#  FIGURA 8 — Osjetljivost na outliere (STROGI 1D — BUKVALNO JEDNA LINIJA)
# ══════════════════════════════════════════════════════════════════════════════
def figura_8_outlier_1D():
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Osjetljivost na outliere — Čista 1D projekcija (bez jitter-a)",
                 fontsize=16, fontweight="bold")

    np.random.seed(42)
    n = 100
    X_clean = np.vstack([
        np.random.randn(n, 2) * 0.5 + [-2, 0],
        np.random.randn(n, 2) * 0.5 + [2, 0],
    ])
    y_out = np.array([0] * n + [1] * n)

    outlier = np.array([[25, 25]])
    X_dirty = np.vstack([X_clean, outlier])
    y_dirty = np.append(y_out, 0)

    for row, (X_data, y_data, label_txt) in enumerate([
        (X_clean, y_out, "BEZ outliera"),
        (X_dirty, y_dirty, "SA outlierom (25, 25)")
    ]):
        # --- 1. Originalni 2D podaci (moraju biti 2D da vidimo outlier) ---
        ax = axes[row, 0]
        for i, color in enumerate(colors_2class):
            mask = y_data == i
            ax.scatter(X_data[mask, 0], X_data[mask, 1], c=color, alpha=0.7, s=40)
        ax.set_title(f"Originalni podaci\n{label_txt}")
        ax.grid(True, alpha=0.3)

        # --- 2. PCA (Strogi 1D) ---
        pca_o = PCA(n_components=1).fit(X_data)
        X_p = pca_o.transform(X_data)
        ax = axes[row, 1]
        for i, color in enumerate(colors_2class):
            mask = y_data == i
            # np.zeros_like pravi niz nula - sve tačke su na istoj visini
            ax.scatter(X_p[mask, 0], np.zeros_like(X_p[mask, 0]),
                       c=color, alpha=0.5, s=50, edgecolors="white", linewidth=0.5)

        ax.set_title(f"PCA (1D)\ninter/intra={inter_intra_ratio(X_p, y_data):.2f}")
        ax.set_ylim(-0.1, 0.1) # Maksimalno sužavamo Y osu
        ax.set_yticks([])      # Brišemo oznake sa Y ose
        ax.axhline(0, color='black', linewidth=1, alpha=0.5) # Crta liniju ose
        ax.grid(True, alpha=0.3, axis='x')

        # --- 3. LDA (Strogi 1D) ---
        lda_o = LDA(n_components=1).fit(X_data, y_data)
        X_l = lda_o.transform(X_data)
        ax = axes[row, 2]
        for i, color in enumerate(colors_2class):
            mask = y_data == i
            ax.scatter(X_l[mask, 0], np.zeros_like(X_l[mask, 0]),
                       c=color, alpha=0.5, s=50, edgecolors="white", linewidth=0.5)

        ax.set_title(f"LDA (1D)\ninter/intra={inter_intra_ratio(X_l, y_data):.2f}")
        ax.set_ylim(-0.1, 0.1)
        ax.set_yticks([])
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig("slike/08_outlier_1D.png", dpi=150, bbox_inches="tight")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — pokreni sve figure i ispiši rezultate
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generisanje figura...\n")

    figura_1_pca_wine()
    print("  ✓ Figura 1: PCA na Wine (01_pca_wine.png)")

    ratio_orig, ratio_pca, ratio_lda = figura_2_lda_wine()
    print("  ✓ Figura 2: LDA na Wine (02_lda_wine.png)")

    figura_3_pca_failures()
    print("  ✓ Figura 3: Podaci koji ruše PCA (03_pca_failures.png)")

    figura_4_lda_failures()
    print("  ✓ Figura 4: Podaci koji ruše LDA (04_lda_failures.png)")

    figura_5_pca_fail_lda_win()
    print("  ✓ Figura 5: PCA fail + LDA win (05_pca_fail_lda_win.png)")

    figura_6_loadings_heatmap()
    print("  ✓ Figura 6: Loadings heatmap (06_loadings_heatmap.png)")

    figura_7_curse_of_dim()
    print("  ✓ Figura 7: Curse of dimensionality (07_curse_of_dim.png)")

    figura_8_outlier_1D()
    print("  ✓ Figura 8: Osjetljivost na outliere — 1D (08_outlier_1D.png)")

    # Numerički izvještaj
    print("\n" + "=" * 60)
    print("  PCA — Objašnjena varijanca po komponentama (Wine)")
    print("=" * 60)
    for i, (ev_i, cev_i) in enumerate(zip(
            pca_full.explained_variance_ratio_,
            np.cumsum(pca_full.explained_variance_ratio_))):
        print(f"  PC{i+1:2d}: {ev_i:6.2%}  (kumulativno: {cev_i:6.2%})")

    print("\n" + "=" * 60)
    print("  LDA — Objašnjena separacija po diskriminantama (Wine)")
    print("=" * 60)
    for i, r in enumerate(lda_full.explained_variance_ratio_):
        print(f"  LD{i+1}: {r:.2%}")

    print("\n" + "=" * 60)
    print("  Separabilnost klasa (Wine dataset)")
    print("=" * 60)
    print(f"  Original (prva 2 feature-a) : {ratio_orig:.4f}")
    print(f"  PCA (2 komponente)          : {ratio_pca:.4f}")
    print(f"  LDA (2 komponente)          : {ratio_lda:.4f}")
    print(f"\n  LDA je {ratio_lda/ratio_pca:.1f}× bolja od PCA u separaciji klasa")
    print("=" * 60)

    #plt.show()
