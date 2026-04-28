
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 2
    vocab_size: int = 30000
    seq_len_max: int = 128

class WeightManager:
    def __init__(self, version: str = "v1"):
        self.base_path = Path(os.path.expanduser("~/hermes-sandbox/ContextWatch/.contextwatch_data"))
        self.weights_path = self.base_path / "weights"
        self.metadata_path = self.base_path / "metadata"
        
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

        self.version = version
        self.encoder_path = self.weights_path / f"encoder_{version}.npy"
        self.attention_path = self.weights_path / f"attention_{version}.npy"
        self.mlkp_head_path = self.weights_path / f"mlkp_{version}.npy"
        self.vhm_center_path = self.weights_path / f"vhm_{version}_center.npy"
        self.vhm_radius_path = self.weights_path / f"vhm_{version}_radius.json"
        self.vhm_volume_path = self.weights_path / f"vhm_{version}_volume.json"

    def has_weights(self) -> bool:
        return all(p.exists() for p in [self.encoder_path, self.attention_path, self.mlkp_head_path, self.vhm_center_path])

    def load_encoder_weights(self) -> Optional[np.ndarray]:
        if not self.encoder_path.exists(): return None
        return np.load(self.encoder_path)

    def load_attention_weights(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.attention_path.exists(): return None
        return np.load(self.attention_path, allow_pickle=True).item()

    def load_mlkp_head(self) -> Optional[np.ndarray]:
        if not self.mlkp_head_path.exists(): return None
        return np.load(self.mlkp_head_path)

    def load_vhm_params(self) -> Optional[Tuple[np.ndarray, float, float]]:
        if not self.vhm_center_path.exists() or not self.vhm_radius_path.exists():
            return None
        center = np.load(self.vhm_center_path)
        with open(self.vhm_radius_path, 'r') as f:
            vhm_data = json.load(f)
        return center, vhm_data["radius"], vhm_data.get("volume", 1.0)

    def save_weights(self, encoder, attention, mlkp, vhm_center, vhm_radius, vhm_volume):
        np.save(self.encoder_path, encoder)
        np.save(self.attention_path, attention)
        np.save(self.mlkp_head_path, mlkp)
        np.save(self.vhm_center_path, vhm_center)
        with open(self.vhm_radius_path, 'w') as f:
            json.dump({"radius": vhm_radius, "volume": vhm_volume}, f)

    def load_config(self) -> ModelConfig:
        return ModelConfig()

    def save_training_metadata(self, **kwargs):
        meta_file = self.metadata_path / f"training_{self.version}.json"
        with open(meta_file, 'w') as f:
            json.dump(kwargs, f)


def get_weight_manager(version: str = 'v1') -> WeightManager:
    return WeightManager(version)
