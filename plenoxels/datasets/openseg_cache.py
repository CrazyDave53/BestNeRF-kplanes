from pathlib import Path

import numpy as np
import torch


def map_pixels_to_feature_pixels(
    x: torch.Tensor,
    y: torch.Tensor,
    rgb_h: int,
    rgb_w: int,
    feat_h: int,
    feat_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_x = torch.div(x * feat_w, rgb_w, rounding_mode="floor")
    feat_y = torch.div(y * feat_h, rgb_h, rounding_mode="floor")
    feat_x = torch.clamp(feat_x, min=0, max=feat_w - 1).long()
    feat_y = torch.clamp(feat_y, min=0, max=feat_h - 1).long()
    return feat_x, feat_y


class OpenSegFeatureCache:
    def __init__(self, root: str | Path, expected_feature_dim: int = 768):
        root = Path(root)
        self.root = self._resolve_root(root)
        self.expected_feature_dim = expected_feature_dim
        self._shards: dict[str, np.ndarray] = {}

    @staticmethod
    def _has_shards(root: Path) -> bool:
        return root.is_dir() and any(root.glob("cam*.npy"))

    @classmethod
    def _resolve_root(cls, root: Path) -> Path:
        direct_candidates = [root, root / "openseg_camckpts"]
        for candidate in direct_candidates:
            if cls._has_shards(candidate):
                return candidate

        matching_roots = []
        if root.exists() and root.is_dir():
            matching_roots.append(root)
        elif root.parent.exists() and root.parent.is_dir():
            matching_roots.extend(
                path for path in root.parent.rglob(root.name) if path.is_dir()
            )

        for matching_root in sorted(set(matching_roots), key=lambda path: str(path)):
            candidates = [matching_root, matching_root / "openseg_camckpts"]
            for candidate in candidates:
                if cls._has_shards(candidate):
                    return candidate

            shard_paths = sorted(matching_root.rglob("cam*.npy"))
            if shard_paths:
                return shard_paths[0].parent

        return root

    def _open_shard(self, camera_name: str) -> np.ndarray:
        shard_name = f"{camera_name}.npy"
        shard_path = self.root / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(
                f"OpenSeg feature shard not found: {shard_name} under {self.root}"
            )

        shard = np.load(shard_path, mmap_mode="r")
        if shard.ndim != 4:
            raise ValueError(
                f"OpenSeg feature shard {shard_name} must have 4 dimensions, got {shard.ndim}"
            )
        if shard.shape[-1] != self.expected_feature_dim:
            raise ValueError(
                f"OpenSeg feature shard {shard_name} has feature dimension {shard.shape[-1]}, "
                f"expected {self.expected_feature_dim}"
            )
        return shard

    def _load_shard(self, camera_name: str) -> np.ndarray:
        if camera_name not in self._shards:
            self._shards[camera_name] = self._open_shard(camera_name)
        return self._shards[camera_name]

    def close(self):
        for shard in self._shards.values():
            mmap = getattr(shard, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._shards.clear()

    def lookup(
        self,
        camera_names: list[str],
        frame_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rgb_h: int,
        rgb_w: int,
    ) -> torch.Tensor:
        if (
            len(camera_names) != frame_ids.numel()
            or len(camera_names) != x.numel()
            or len(camera_names) != y.numel()
        ):
            raise ValueError("camera_names, frame_ids, x, and y must describe the same batch size")

        features = np.empty((len(camera_names), self.expected_feature_dim), dtype=np.float16)
        frame_ids_cpu = frame_ids.detach().cpu()
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for camera_name in dict.fromkeys(camera_names):
            shard = self._load_shard(camera_name)
            frame_count, feat_h, feat_w, _ = shard.shape
            indices = [idx for idx, name in enumerate(camera_names) if name == camera_name]
            index_tensor = torch.tensor(indices, dtype=torch.long)
            camera_frame_ids = frame_ids_cpu[index_tensor].long()
            if (camera_frame_ids < 0).any() or (camera_frame_ids >= frame_count).any():
                bad_index = int(
                    ((camera_frame_ids < 0) | (camera_frame_ids >= frame_count))
                    .nonzero()[0]
                    .item()
                )
                bad_frame = int(camera_frame_ids[bad_index].item())
                raise IndexError(
                    f"OpenSeg feature lookup for camera {camera_name} got frame id "
                    f"{bad_frame}; valid frame count is {frame_count}"
                )

            feat_x, feat_y = map_pixels_to_feature_pixels(
                x=x_cpu[index_tensor],
                y=y_cpu[index_tensor],
                rgb_h=rgb_h,
                rgb_w=rgb_w,
                feat_h=feat_h,
                feat_w=feat_w,
            )
            features[indices] = shard[
                camera_frame_ids.numpy(),
                feat_y.numpy(),
                feat_x.numpy(),
            ]
        return torch.from_numpy(features)
