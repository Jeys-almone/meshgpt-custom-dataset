from pathlib import Path
import trimesh

input_dir = Path("meshes")
output_dir = Path("meshes_simplified")
output_dir.mkdir(exist_ok=True)

target_faces = 250

mesh_paths = sorted(input_dir.rglob("*.obj"))

for path in mesh_paths:
    mesh = trimesh.load(path, force="mesh")

    print(f"\nProcessando: {path}")
    print(f"Faces originais: {len(mesh.faces)}")

    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

    relative = path.relative_to(input_dir)
    out_path = output_dir / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(out_path)

    print(f"Salvo em: {out_path}")
    print(f"Faces finais: {len(mesh.faces)}")