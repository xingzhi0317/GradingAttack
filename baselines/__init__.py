from .gcg import *

try:
    from .roleplay import *
except Exception:
    # RolePlay depends on optional vLLM; keep GCG usable without it.
    pass
