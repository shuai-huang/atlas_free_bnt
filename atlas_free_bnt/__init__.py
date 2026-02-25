"""Atlas-free Brain Network Transformer (atlas-free BNT).

This package provides:
- Model: AtlasFreeBNT
- Dataset: AtlasFreeDataset
- Utilities: logging, seeding, MAT loading, block coordinate generation
"""

from .af_bnt import AtlasFreeBNT
from .dataset import AtlasFreeDataset, make_dataloaders
from .af_bnt import AtlasFreeBNT, ModelConfig
from .dataset import AtlasFreeDataset, DatasetConfig
from .utils import BlockSpec
