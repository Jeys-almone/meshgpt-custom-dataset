from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    mesh = trimesh.load(path, force="mesh")

    if mesh is None:
        raise ValueError(f"Falha ao carregar malha: {path}")

    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise ValueError(f"Malha sem vértices: {path}")

    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError(f"Malha sem faces: {path}")

    if mesh.is_empty:
        raise ValueError(f"Malha vazia: {path}")

    return mesh


def normalize_mesh(mesh):
    mesh = mesh.copy()

    vertices = mesh.vertices.astype(np.float64)

    center = vertices.mean(axis=0)
    vertices -= center

    scale = np.max(np.linalg.norm(vertices, axis=1))
    if scale > 0:
        vertices /= scale

    mesh.vertices = vertices
    return mesh


def sample_points(mesh, n_points=2048):
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def chamfer_distance(points_a, points_b):
    from scipy.spatial import cKDTree

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a, _ = tree_b.query(points_a)
    dist_b, _ = tree_a.query(points_b)

    return float(np.mean(dist_a**2) + np.mean(dist_b**2))


def mesh_stats(mesh):
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces))
    }