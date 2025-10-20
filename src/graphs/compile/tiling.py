# characterize/tiling.py

class TilingStrategy:
    def compute_tile_count(self, op_type, op_metadata, kernel_params=None):
        raise NotImplementedError("Override in subclass")

class CPUTilingStrategy(TilingStrategy):
    def compute_tile_count(self, op_type, op_metadata, kernel_params=None):
        # Assume 256KB usable per tile, 3 buffers
        tile_mem = 256 * 1024
        shape = op_metadata.get("input_shape", [32, 128])
        total_bytes = sum([dim * 4 for dim in shape]) * 3
        return max(1, total_bytes // tile_mem)

class GPUTilingStrategy(TilingStrategy):
    def compute_tile_count(self, op_type, op_metadata, kernel_params=None):
        tile_mem = 48 * 1024
        shape = op_metadata.get("input_shape", [32, 128])
        total_bytes = sum([dim * 4 for dim in shape])
        return max(1, total_bytes // tile_mem)

class TPUTilingStrategy(TilingStrategy):
    def compute_tile_count(self, op_type, op_metadata, kernel_params=None):
        tile_mem = 24 * 1024 * 1024
        shape = op_metadata.get("input_shape", [32, 128])
        total_bytes = sum([dim * 4 for dim in shape])
        return max(1, total_bytes // tile_mem)

class KPUTilingStrategy(TilingStrategy):
    def compute_tile_count(self, op_type, op_metadata, kernel_params=None):
        shape = op_metadata.get("input_shape", [32, 128])
        N = shape[-1] if shape else 1
        wavefront_mem = N * N * 4
        return 1 if wavefront_mem < 64 * 1024 * 1024 else 2
