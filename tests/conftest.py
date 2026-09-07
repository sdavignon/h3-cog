import sys
import types


if sys.platform == "win32":
    sys.modules.setdefault(
        "fcntl",
        types.SimpleNamespace(LOCK_EX=1, flock=lambda *_: None),
    )
