from pathlib import Path
import torch
import trimesh
from meshgpt_pytorch import MeshAutoencoder

obj_path = Path("meshes/Leao/Leao.obj")

mesh = trimesh.load(obj_path, force="mesh")

vertices = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0)
faces = torch.tensor(mesh.faces, dtype=torch.long).unsqueeze(0)

print("vertices:", vertices.shape)
print("faces:", faces.shape)

autoencoder = MeshAutoencoder(
    num_discrete_coors=128
)

print("Autoencoder criado com sucesso!")
print("Entrada pronta para o MeshGPT.")