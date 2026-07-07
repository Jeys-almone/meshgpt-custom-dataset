from pathlib import Path
from rembg import remove
from PIL import Image

# Pasta principal onde estão as subpastas:
# dataset/ArturAzevedoBusto
# dataset/fotos_canho
# dataset/Leao
# dataset/Postes
INPUT_DIR = Path("dataset")

# Pasta onde serão salvas as imagens sem fundo
OUTPUT_DIR = Path("dataset_sem_fundo")

EXTENSOES = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

images = []

for ext in EXTENSOES:
    images.extend(INPUT_DIR.rglob(ext))

print(f"Total de imagens encontradas: {len(images)}")

if len(images) == 0:
    raise RuntimeError("Nenhuma imagem encontrada dentro da pasta dataset")

for img_path in images:
    try:
        print(f"Processando: {img_path}")

        # Mantém a mesma estrutura de subpastas na saída
        relative_path = img_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path.with_suffix(".png")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(img_path).convert("RGBA")
        image_sem_fundo = remove(image)

        image_sem_fundo.save(output_path)

        print(f"Salvo em: {output_path}")

    except Exception as e:
        print(f"Erro ao processar {img_path}: {e}")

print("Remoção de fundo concluída!")