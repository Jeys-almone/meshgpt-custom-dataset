import open3d as o3d
from pathlib import Path

input_dir = Path("evaluation/point_e_ply")
output_dir = Path("evaluation/point_e")
output_dir.mkdir(parents=True, exist_ok=True)

for ply_path in input_dir.glob("*.ply"):
    print(f"Processando: {ply_path.name}")

    pcd = o3d.io.read_point_cloud(str(ply_path))

    if pcd.is_empty():
        print(f"[AVISO] Nuvem vazia: {ply_path}")
        continue

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=8
    )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    output_path = output_dir / f"{ply_path.stem}.obj"
    o3d.io.write_triangle_mesh(str(output_path), mesh)

    print(f"Salvo: {output_path}")

print("Conversão concluída!")
