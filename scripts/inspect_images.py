from pathlib import Path
from PIL import Image

dataset = Path("dataset")

for categoria in sorted(dataset.iterdir()):
    if categoria.is_dir():
        print(f"\nCategoria: {categoria.name}")

        for img_path in categoria.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                with Image.open(img_path) as img:
                    print(f"{img_path.name} | tamanho={img.size} | modo={img.mode}")