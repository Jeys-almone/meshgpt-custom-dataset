import torch
from pathlib import Path
from datetime import datetime
from meshgpt_pytorch import MeshAutoencoder, MeshTransformer

# =====================================================
# DISPOSITIVO
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# =====================================================
# PASTA DE SAÍDA
# =====================================================

Path("generated_meshes").mkdir(exist_ok=True)

# =====================================================
# AUTOENCODER
# =====================================================

autoencoder = MeshAutoencoder(
    num_discrete_coors=128
).to(device)

autoencoder.load_state_dict(
    torch.load(
        "models/autoencoder_all_meshes.pt",
        map_location=device
    )
)

autoencoder.eval()

print("Autoencoder geral carregado!")

# =====================================================
# TRANSFORMER
# =====================================================

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

transformer.load_state_dict(
    torch.load(
        "models/transformer_all_meshes.pt",
        map_location=device
    )
)

transformer.eval()

print("Transformer geral carregado!")

# =====================================================
# GERAÇÃO
# =====================================================

with torch.no_grad():
    face_coords, face_mask = transformer.generate(
        batch_size=1,
        temperature=0.8
    )

face_coords = face_coords[0].cpu()
face_mask = face_mask[0].cpu().bool()

face_coords = face_coords[face_mask]

# =====================================================
# EXPORTAÇÃO
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
obj_path = f"generated_meshes/generated_mesh_{timestamp}.obj"

with open(obj_path, "w") as f:

    vertex_index = 1

    for triangle in face_coords:

        for vertex in triangle:
            x, y, z = vertex.tolist()
            f.write(f"v {x} {y} {z}\n")

        f.write(
            f"f {vertex_index} {vertex_index + 1} {vertex_index + 2}\n"
        )

        vertex_index += 3

print("\n===================================")
print("Malha gerada com sucesso!")
print(f"Arquivo salvo em: {obj_path}")
print(f"Total de triângulos: {len(face_coords)}")
print("===================================")