from pathlib import Path
import pandas as pd

from utils import (
    load_mesh,
    normalize_mesh,
    sample_points,
    mesh_stats
)

from metrics import (
    chamfer_distance,
    compute_mmd,
    compute_cov,
    compute_1nna
)


REAL_DIR = Path("evaluation/real")

METHOD_DIRS = {
    "MeshGPT": Path("evaluation/meshgpt"),
    "Point-E": Path("evaluation/point_e"),
    "Shap-E": Path("evaluation/shap_e"),
}

OUTPUT_DIR = Path("evaluation/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_POINTS = 2048


# =====================================================
# CARREGA MALHAS REAIS
# =====================================================

real_point_sets = []
real_names = []

for path in sorted(REAL_DIR.glob("*.obj")):
    try:
        mesh = normalize_mesh(load_mesh(path))
        points = sample_points(mesh, N_POINTS)

        real_point_sets.append(points)
        real_names.append(path.name)

    except Exception as e:
        print(f"[AVISO] Malha real ignorada: {path}")
        print(f"Motivo: {e}")

print(f"Malhas reais carregadas: {len(real_point_sets)}")

if len(real_point_sets) == 0:
    raise RuntimeError("Nenhuma malha real válida foi carregada.")


# =====================================================
# AVALIA CADA MÉTODO
# =====================================================

detailed_rows = []
summary_rows = []

for method_name, method_dir in METHOD_DIRS.items():

    generated_paths = sorted(method_dir.glob("*.obj"))

    if len(generated_paths) == 0:
        print(f"[AVISO] Nenhuma malha encontrada para {method_name} em {method_dir}")
        continue

    generated_point_sets = []
    vertices_list = []
    faces_list = []
    valid_count = 0
    invalid_count = 0

    print(f"\nAvaliando método: {method_name}")

    for gen_path in generated_paths:

        try:
            gen_mesh = normalize_mesh(load_mesh(gen_path))
            gen_points = sample_points(gen_mesh, N_POINTS)
            stats = mesh_stats(gen_mesh)

        except Exception as e:
            invalid_count += 1
            print(f"[AVISO] Malha gerada ignorada: {gen_path}")
            print(f"Motivo: {e}")
            continue

        valid_count += 1

        generated_point_sets.append(gen_points)
        vertices_list.append(stats["vertices"])
        faces_list.append(stats["faces"])

        for real_name, real_points in zip(real_names, real_point_sets):
            cd = chamfer_distance(gen_points, real_points)

            detailed_rows.append({
                "method": method_name,
                "generated_mesh": gen_path.name,
                "reference_mesh": real_name,
                "chamfer_distance": cd,
                "vertices": stats["vertices"],
                "faces": stats["faces"]
            })

    if valid_count == 0:
        print(f"[AVISO] Nenhuma malha válida para {method_name}. Pulando métricas.")
        summary_rows.append({
            "method": method_name,
            "chamfer_mean": None,
            "chamfer_min": None,
            "chamfer_max": None,
            "mmd": None,
            "cov": None,
            "one_nna": None,
            "vertices_mean": None,
            "faces_mean": None,
            "num_generated_meshes": len(generated_paths),
            "num_valid_meshes": valid_count,
            "num_invalid_meshes": invalid_count
        })
        continue

    chamfer_values = [
        row["chamfer_distance"]
        for row in detailed_rows
        if row["method"] == method_name
    ]

    mmd = compute_mmd(real_point_sets, generated_point_sets)
    cov = compute_cov(real_point_sets, generated_point_sets)
    nna = compute_1nna(real_point_sets, generated_point_sets)

    summary_rows.append({
        "method": method_name,
        "chamfer_mean": sum(chamfer_values) / len(chamfer_values),
        "chamfer_min": min(chamfer_values),
        "chamfer_max": max(chamfer_values),
        "mmd": mmd,
        "cov": cov,
        "one_nna": nna,
        "vertices_mean": sum(vertices_list) / len(vertices_list),
        "faces_mean": sum(faces_list) / len(faces_list),
        "num_generated_meshes": len(generated_paths),
        "num_valid_meshes": valid_count,
        "num_invalid_meshes": invalid_count
    })

    print(f"MMD: {mmd:.6f}")
    print(f"COV: {cov:.6f}")
    print(f"1-NNA: {nna:.6f}")
    print(f"Vértices médios: {sum(vertices_list) / len(vertices_list):.2f}")
    print(f"Faces médias: {sum(faces_list) / len(faces_list):.2f}")
    print(f"Malhas válidas: {valid_count}")
    print(f"Malhas inválidas: {invalid_count}")


# =====================================================
# SALVA RESULTADOS
# =====================================================

df_detailed = pd.DataFrame(detailed_rows)
df_summary = pd.DataFrame(summary_rows)

detailed_path = OUTPUT_DIR / "comparison_results.csv"
summary_path = OUTPUT_DIR / "comparison_summary.csv"

df_detailed.to_csv(detailed_path, index=False)
df_summary.to_csv(summary_path, index=False)

latex_path = OUTPUT_DIR / "comparison_table.tex"
df_summary.to_latex(latex_path, index=False, float_format="%.6f")


print("\n===================================")
print("Comparação concluída!")
print(f"Resultados detalhados: {detailed_path}")
print(f"Resumo por método: {summary_path}")
print(f"Tabela LaTeX: {latex_path}")
print("===================================")