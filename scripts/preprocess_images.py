from pathlib import Path
from PIL import Image

entrada = Path("dataset")
saida = Path("dataset_preprocessado")
tamanho = (512, 512)

saida.mkdir(exist_ok=True)

for categoria in entrada.iterdir():
    if categoria.is_dir():
        pasta_saida = saida / categoria.name
        pasta_saida.mkdir(exist_ok=True)

        contador = 1

        for img_path in categoria.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail(tamanho)

                    canvas = Image.new("RGB", tamanho, (255, 255, 255))
                    x = (tamanho[0] - img.width) // 2
                    y = (tamanho[1] - img.height) // 2
                    canvas.paste(img, (x, y))

                    nome_saida = f"{categoria.name}_{contador:03d}.jpg"
                    canvas.save(pasta_saida / nome_saida, quality=95)

                    contador += 1

print("Pré-processamento concluído!")