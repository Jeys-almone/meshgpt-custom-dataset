from pathlib import Path

dataset = Path("dataset")

total = 0

for categoria in sorted(dataset.iterdir()):
    if categoria.is_dir():
        imagens = [
            img for img in categoria.iterdir()
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{categoria.name}: {len(imagens)} imagens")

        total += len(imagens)

print(f"\nTotal de imagens: {total}")