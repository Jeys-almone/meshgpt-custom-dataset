print("INICIOU GERAÇÃO POINT-E", flush=True)

from pathlib import Path
from PIL import Image
import torch
import numpy as np

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint


DATASET_DIR = Path("dataset_sem_fundo")
OUTPUT_DIR = Path("evaluation/point_e")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

categories = [
    "ArturAzevedoBusto",
    "fotos_canho",
    "Leao",
    "Postes",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device, flush=True)


def get_first_image(folder):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

    images = []
    for ext in extensions:
        images.extend(folder.glob(ext))

    images = sorted(images)

    if not images:
        return None

    return images[0]


def save_point_cloud_as_ply(pc, output_file):
    coords = pc.coords
    channels = pc.channels

    r = channels["R"]
    g = channels["G"]
    b = channels["B"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {coords.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(coords.shape[0]):
            x, y, z = coords[i]
            red = int(np.clip(r[i], 0, 255))
            green = int(np.clip(g[i], 0, 255))
            blue = int(np.clip(b[i], 0, 255))
            f.write(f"{x} {y} {z} {red} {green} {blue}\n")


print("Carregando modelo Point-E...", flush=True)

base_name = "base40M-textvec"

base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()

base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
base_model.load_state_dict(load_checkpoint(base_name, device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model],
    diffusions=[base_diffusion],
    num_points=[1024],
    aux_channels=["R", "G", "B"],
    model_kwargs_key_filter=["images"],
    guidance_scale=[3.0],
    clip_denoised=True,
    use_karras=[True],
    karras_steps=[16],
    sigma_min=[0.001],
    sigma_max=[120],
    s_churn=[3],
)

print("Modelo carregado!", flush=True)

for category in categories:
    print("\n===================================", flush=True)
    print(f"Categoria: {category}", flush=True)

    category_dir = DATASET_DIR / category
    image_path = get_first_image(category_dir)

    if image_path is None:
        print(f"Nenhuma imagem encontrada em {category_dir}", flush=True)
        continue

    print(f"Imagem usada: {image_path}", flush=True)

    image = Image.open(image_path).convert("RGB").resize((128, 128))

    print("Gerando nuvem de pontos...", flush=True)

    samples = None

    for sample in sampler.sample_batch_progressive(
        batch_size=1,
        model_kwargs=dict(images=[image]),
    ):
        samples = sample
        print("Passo concluído...", flush=True)

    print("Convertendo para point cloud...", flush=True)

    pc = sampler.output_to_point_clouds(samples)[0]

    output_file = OUTPUT_DIR / f"{category}.ply"
    save_point_cloud_as_ply(pc, output_file)

    print(f"Arquivo salvo em: {output_file}", flush=True)

print("\nFINALIZADO!", flush=True)