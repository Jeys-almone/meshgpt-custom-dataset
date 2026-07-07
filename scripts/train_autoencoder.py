from pathlib import Path
import sys

import torch
import trimesh

sys.path.append(str(Path(__file__).resolve().parents[1]))

from meshgpt_pytorch import MeshAutoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

mesh_paths = sorted(Path("meshes_augmented").rglob("*.obj"))

print(f"Total de malhas encontradas: {len(mesh_paths)}")
for path in mesh_paths:
    print("-", path)

if len(mesh_paths) == 0:
    raise RuntimeError("Nenhuma malha .obj encontrada em meshes_augmented")

autoencoder = MeshAutoencoder(
    num_discrete_coors=128
).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

epochs = 20

for epoch in range(epochs):
    autoencoder.train()
    total_loss = 0

    for obj_path in mesh_paths:
        mesh = trimesh.load(obj_path, force="mesh")

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0).to(device)
        faces = torch.tensor(mesh.faces, dtype=torch.long).unsqueeze(0).to(device)

        optimizer.zero_grad()

        loss = autoencoder(
            vertices=vertices,
            faces=faces
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    media_loss = total_loss / len(mesh_paths)

    print(f"Época {epoch + 1}/{epochs} | Loss média: {media_loss:.4f}")

Path("models").mkdir(exist_ok=True)
torch.save(autoencoder.state_dict(), "models/autoencoder_all_meshes.pt")

print("Treinamento concluído!")
print("Modelo salvo em models/autoencoder_all_meshes.pt")