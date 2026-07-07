import numpy as np
from scipy.spatial import cKDTree


# ==========================================================
# Chamfer Distance
# ==========================================================

def chamfer_distance(points_a, points_b):
    """
    Calcula a Chamfer Distance entre duas nuvens de pontos.
    """

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a, _ = tree_b.query(points_a)
    dist_b, _ = tree_a.query(points_b)

    return np.mean(dist_a ** 2) + np.mean(dist_b ** 2)


# ==========================================================
# Minimum Matching Distance (MMD)
# ==========================================================

def compute_mmd(real_point_sets, generated_point_sets):

    distances = []

    for generated in generated_point_sets:

        cds = []

        for real in real_point_sets:
            cds.append(chamfer_distance(real, generated))

        distances.append(min(cds))

    return float(np.mean(distances))


# ==========================================================
# Coverage (COV)
# ==========================================================

def compute_cov(real_point_sets, generated_point_sets):

    matched = set()

    for generated in generated_point_sets:

        cds = [
            chamfer_distance(real, generated)
            for real in real_point_sets
        ]

        matched.add(np.argmin(cds))

    return len(matched) / len(real_point_sets)


# ==========================================================
# 1-NNA
# ==========================================================

def compute_1nna(real_point_sets, generated_point_sets):

    correct = 0
    total = len(real_point_sets) + len(generated_point_sets)

    all_sets = real_point_sets + generated_point_sets

    labels = (
        [0] * len(real_point_sets) +
        [1] * len(generated_point_sets)
    )

    for i in range(total):

        nearest = None
        best = np.inf

        for j in range(total):

            if i == j:
                continue

            d = chamfer_distance(
                all_sets[i],
                all_sets[j]
            )

            if d < best:
                best = d
                nearest = j

        if labels[i] == labels[nearest]:
            correct += 1

    return correct / total