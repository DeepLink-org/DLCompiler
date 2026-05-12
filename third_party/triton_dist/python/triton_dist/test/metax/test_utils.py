from triton_dist.kernels.metax.utils import get_numa_node, has_fullmesh_mxlink

full_mesh = has_fullmesh_mxlink()
numa_node = get_numa_node(0)
print(full_mesh)
print(numa_node)
