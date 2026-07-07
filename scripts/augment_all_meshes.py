import os
import trimesh
import numpy as np

INPUT_DIR = "meshes_simplified"
OUTPUT_DIR = "meshes_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_VARIATIONS = 50

for root, dirs, files in os.walk(INPUT_DIR):

    for file in files:

        if not file.lower().endswith(".obj"):
            continue

        input_path = os.path.join(root, file)

        rel_dir = os.path.relpath(root, INPUT_DIR)
        out_dir = os.path.join(OUTPUT_DIR, rel_dir)

        os.makedirs(out_dir, exist_ok=True)

        mesh = trimesh.load(input_path, force="mesh")

        name = os.path.splitext(file)[0]

        # salva a original
        mesh.export(os.path.join(out_dir, f"{name}_000.obj"))

        for i in range(1, N_VARIATIONS):

            m = mesh.copy()

            # rotação aleatória
            angle = np.random.uniform(0, 360)

            axis = np.random.normal(size=3)
            axis /= np.linalg.norm(axis)

            matrix = trimesh.transformations.rotation_matrix(
                np.radians(angle),
                axis
            )

            m.apply_transform(matrix)

            # pequena escala
            scale = np.random.uniform(0.95, 1.05)
            m.apply_scale(scale)

            # pequena translação
            translation = np.random.uniform(-0.02, 0.02, 3)
            m.apply_translation(translation)

            # pequeno ruído
            noise = np.random.normal(
                0,
                0.001,
                m.vertices.shape
            )

            m.vertices += noise

            m.export(
                os.path.join(
                    out_dir,
                    f"{name}_{i:03d}.obj"
                )
            )

print("Dataset aumentado criado com sucesso!")