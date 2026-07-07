from pathlib import Path
import csv

dataset = Path("dataset_preprocessado")
saida = Path("manifest_dataset.csv")

with open(saida, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["categoria", "imagem"])

    for categoria in sorted(dataset.iterdir()):
        if categoria.is_dir():
            for img in sorted(categoria.glob("*.jpg")):
                writer.writerow([categoria.name, str(img)])

print(f"Manifesto criado: {saida}")