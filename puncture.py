import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter, minimum_filter

# Configuration
BASE = Path("Output/rk4/bowen_york_minimal")
OUT = BASE / "analysis"
OUT.mkdir(exist_ok=True, parents=True)

# Récupération des dossiers triés (step1, step2, ... step10)
# On trie par numéro pour éviter l'ordre alphabétique (step1, step10, step2...)


def get_step_number(p):
    try:
        return int(p.name.replace("step", ""))
    except ValueError:
        return -1


STEPS = sorted([d for d in BASE.iterdir() if d.is_dir()
               and "step" in d.name], key=get_step_number)

# Fichier de log
log_path = OUT / "punctures.csv"
puncture_log = open(log_path, "w")
puncture_log.write("step,x1,y1,x2,y2\n")


def load_slice_robust(path):
    """
    Charge le CSV et tente de le transformer en grille 2D (Nx, Ny).
    """
    try:
        # skiprows=1 pour sauter l'en-tête
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"[ERR] Lecture impossible de {path}: {e}")
        return None

    if data.size == 0:
        return None

    # CAS 1 : Le fichier est déjà une matrice carrée (NxN valeurs)
    # Vérification simple : est-ce que c'est carré ?
    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        return data

    # CAS 2 : Le fichier est en colonnes (x, y, val) ou (x, y, z, val)
    # C'est souvent le cas si data.shape est (N_points, 3) ou (N_points, 4)
    if data.ndim == 2 and data.shape[1] >= 3:
        # On suppose que c'est la dernière colonne qui contient Chi
        values = data[:, -1]

        # On essaie de trouver N tel que N*N = nombre de lignes
        size = values.size
        N = int(np.sqrt(size))

        if N * N == size:
            # On reshape. Attention à l'ordre (C-order vs Fortran-order).
            # En général, les boucles C++ (i,j) remplissent ligne par ligne -> reshape(N, N)
            return values.reshape(N, N)

    print(
        f"[WARN] Format de données non reconnu pour {path}. Shape: {data.shape}")
    return None


def find_punctures(chi, n=2):
    """Trouve les n minima locaux les plus profonds (trous noirs)"""
    # 1. Lissage pour éviter de détecter du bruit de grille comme un trou noir
    sm = gaussian_filter(chi, sigma=2.0)

    # 2. Filtre de minimum local (taille 15 = rayon de recherche ~7 points)
    mins = minimum_filter(sm, size=15)

    # 3. Masque booléen là où la valeur lissée == le minimum local
    mask = (sm == mins)

    # 4. Extraction des coordonnées
    ys, xs = np.where(mask)
    vals = sm[ys, xs]

    # 5. Tri pour garder les trous noirs les plus "profonds" (valeur la plus faible)
    # Chi ~ 0 au niveau de l'horizon
    order = np.argsort(vals)

    # On garde les n premiers
    final_xs = xs[order][:n]
    final_ys = ys[order][:n]

    return list(zip(final_xs, final_ys))

# --- Boucle Principale ---


print(f"Analyse de {len(STEPS)} étapes...")

for step_dir in STEPS:
    step_name = step_dir.name
    chi_path = step_dir / "chi_z.csv"

    if not chi_path.exists():
        continue

    chi = load_slice_robust(chi_path)

    if chi is None:
        continue

    # Recherche
    punctures = find_punctures(chi, n=2)

    # Plot
    plt.figure(figsize=(6, 5))
    # origin="lower" est important pour que (0,0) soit en bas à gauche (physique)
    plt.imshow(chi, origin="lower", cmap="inferno", interpolation="nearest")
    plt.colorbar(label=r"$\psi^{-4}$ aka $\chi$")

    coords_str = []
    for i, (x, y) in enumerate(punctures):
        # x correspond aux colonnes, y aux lignes dans imshow
        plt.scatter(x, y, c="cyan", s=80, marker="x", linewidths=2)
        plt.text(x + 3, y + 3, f"BH{i+1}",
                 color="cyan", fontsize=9, fontweight='bold')
        coords_str.append(f"({x}, {y})")

    plt.title(f"{step_name} : {', '.join(coords_str)}")
    plt.tight_layout()

    out_img = OUT / f"{step_name}.png"
    plt.savefig(out_img, dpi=100)
    plt.close()

    # Logging si on a trouvé les 2
    if len(punctures) == 2:
        (x1, y1), (x2, y2) = punctures
        puncture_log.write(f"{step_name},{x1},{y1},{x2},{y2}\n")
        print(f" -> {step_name}: BH1({x1},{y1}) BH2({x2},{y2})")
    else:
        print(
            f" -> {step_name}: Trouvé {len(punctures)} puncture(s) (attendu: 2)")

puncture_log.close()
print(f"Terminé. Résultats dans {OUT}")
