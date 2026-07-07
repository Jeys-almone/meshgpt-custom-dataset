from pathlib import Path
import sys

import torch
import trimesh
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from meshgpt_pytorch import MeshAutoencoder, MeshTransformer, MeshTransformerTrainer


class ObjMeshDataset(Dataset):
    def __init__(self, folder):
        self.mesh_paths = sorted(Path(folder).rglob("*.obj"))

        print(f"Total de malhas encontradas: {len(self.mesh_paths)}")
        for path in self.mesh_paths:
            print("-", path)

        if len(self.mesh_paths) == 0:
            raise RuntimeError(f"Nenhuma malha .obj encontrada em {folder}")

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        obj_path = self.mesh_paths[idx]
        mesh = trimesh.load(obj_path, force="mesh")

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)

        return {
            "vertices": vertices,
            "faces": faces
        }


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

dataset = ObjMeshDataset("meshes_augmented")

autoencoder = MeshAutoencoder(
    num_discrete_coors=128
).to(device)

autoencoder.load_state_dict(
    torch.load("models/autoencoder_all_meshes.pt", map_location=device)
)

autoencoder.eval()
print("Autoencoder carregado!")

transformer = MeshTransformer(
    autoencoder=autoencoder,
    dim=256,
    max_seq_len=1512,
    flash_attn=False,
    attn_depth=4,
    attn_heads=8,
    fine_attn_depth=2,
    fine_attn_heads=8
).to(device)

trainer = MeshTransformerTrainer(
    model=transformer,
    dataset=dataset,
    num_train_steps=5000,
    batch_size=1,
    grad_accum_every=1,
    learning_rate=1e-4,
    checkpoint_every=500,
    checkpoint_folder="models/transformer_checkpoints",
    data_kwargs=["vertices", "faces"]
)

trainer.train()

Path("models").mkdir(exist_ok=True)
torch.save(transformer.state_dict(), "models/transformer_all_meshes.pt")

print("Treinamento do Transformer concluído!")
print("Modelo salvo em models/transformer_all_meshes.pt")