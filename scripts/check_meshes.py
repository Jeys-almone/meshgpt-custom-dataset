from pathlib import Path
import trimesh

meshes_path = Path("meshes")

for obj in meshes_path.rglob("*.obj"):
    mesh = trimesh.load(obj, force="mesh")

    print(f"\nArquivo: {obj}")
    print(f"Vértices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")
    