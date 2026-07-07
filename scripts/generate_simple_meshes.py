from pathlib import Path
import numpy as np
from PIL import Image
import trimesh

entrada = Path("dataset_preprocessado")
saida = Path("meshes")
saida.mkdir(exist_ok=True)

for categoria in sorted(entrada.iterdir()):
    if not categoria.is_dir():
        continue

    imagens = sorted(categoria.glob("*.jpg"))
    if not imagens:
        continue

    img_path = imagens[0]
    print(f"Gerando malha para {categoria.name} usando {img_path.name}")

    img = Image.open(img_path).convert("L").resize((64, 64))
    arr = np.array(img) / 255.0

    vertices = []
    faces = []

    h, w = arr.shape

    for y in range(h):
        for x in range(w):
            z = arr[y, x] * 0.3
            vertices.append([x / w, y / h, z])

    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            faces.append([i, i + 1, i + w])
            faces.append([i + 1, i + w + 1, i + w])

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    pasta_saida = saida / categoria.name
    pasta_saida.mkdir(exist_ok=True)

    mesh.export(pasta_saida / f"{categoria.name}.obj")

print("Malhas geradas com sucesso!")