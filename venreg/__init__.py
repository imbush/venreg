"""venreg — landmark-driven 3D registration of venation (vessel-network) volumes.

Works on any pair of TIFF z-stacks. Registration is derived purely from corresponding
landmark points (no voxel-size assumptions); it searches both proper and reflected
(mirror) similarity, which was the key to aligning samples flipped during processing.
"""
__version__ = "0.1.0"
