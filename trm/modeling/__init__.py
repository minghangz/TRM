from .trm import TRM
ARCHITECTURES = {"TRM": TRM}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
