import os
import torch
from pathlib import Path
from PIL import Image
from typing import Union
import numpy as np

# Importações específicas do Shap-E
try:
    from shap_e.models.download import load_model, load_config
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
    from shap_e.models.transmitter.base import Transmitter, VectorDecoder
    from shap_e.rendering.torch_mesh import TorchMesh
    from shap_e.util.collections import AttrDict
except ImportError:
    print("Erro: A biblioteca 'shap-e' não está instalada.")
    print("Por favor, instale-a primeiro usando:")
    print("pip install git+https://github.com/openai/shap-e.git")
    exit(1)

# Recriamos a função decode_latent_mesh do util.notebooks para evitar a importação do ipywidgets
def create_pan_cameras(size: int, device: torch.device) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )

@torch.no_grad()
def decode_latent_mesh(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
) -> TorchMesh:
    decoded = xm.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, latent.device)),  # lowest resolution possible
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]

DATASET_DIR = Path("dataset_sem_fundo")
OUTPUT_DIR = Path("evaluation/shap_e")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

categories = [
    "ArturAzevedoBusto",
    "fotos_canho",
    "Leao",
    "Postes",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo detectado:", device, flush=True)

def get_first_image(folder):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    images = []
    for ext in extensions:
        images.extend(folder.glob(ext))
    images = sorted(images)
    if not images:
        return None
    return images[0]

print("Carregando modelos do Shap-E...", flush=True)
xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
print("Modelos carregados com sucesso!", flush=True)


for category in categories:
    print("\n===================================", flush=True)
    print(f"Processando categoria: {category}", flush=True)
    
    category_dir = DATASET_DIR / category
    image_path = get_first_image(category_dir)
    
    if image_path is None:
        print(f"Nenhuma imagem encontrada em {category_dir}", flush=True)
        continue
        
    print(f"Imagem de entrada: {image_path}", flush=True)
    image = Image.open(image_path).convert("RGB")
    
    # Executa a geração usando Shap-E
    print("Gerando latents usando Shap-E...", flush=True)
    
    # Para placas com pouca VRAM (como a GTX 1650 de 4GB), limpamos o cache antes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=3.0,
        model_kwargs=dict(images=[image]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    
    latent = latents[0]
    
    # Decodifica os latentes para uma malha triangular (tri_mesh)
    print("Decodificando latents para malha triangular...", flush=True)
    t = decode_latent_mesh(xm, latent).tri_mesh()
    
    # Define o nome da malha gerada
    # Usaremos o mesmo nome da categoria
    obj_file = OUTPUT_DIR / f"{category}.obj"
    ply_file = OUTPUT_DIR / f"{category}.ply"
    
    print(f"Salvando malha em {obj_file}...", flush=True)
    with open(obj_file, "w", encoding="utf-8") as f:
        t.write_obj(f)
        
    print(f"Salvando malha em {ply_file}...", flush=True)
    with open(ply_file, "wb") as f:
        t.write_ply(f)
        
    print(f"Categoria {category} concluída!", flush=True)

print("\n===================================")
print("Geração do Shap-E concluída com sucesso!")
print("Arquivos salvos em:", OUTPUT_DIR)
print("===================================")
