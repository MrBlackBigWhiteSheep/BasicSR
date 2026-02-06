from basicsr.models.mambawater_model import MambaWaterModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VmambaModel(MambaWaterModel):
    """Vmamba model wrapper (uses MambaWater training logic with Mamber32 arch)."""

    pass
