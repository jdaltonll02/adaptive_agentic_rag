from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

# Re-export baseline verl public API so `from verl import DataProto` keeps working.
from verl.protocol import DataProto  # noqa: F401

try:
    import os as _os
    _vf = _os.path.join(_os.path.dirname(__file__), '..', 'baseline', 'verl', 'version', 'version')
    with open(_os.path.normpath(_vf)) as _fh:
        __version__ = _fh.read().strip()
except Exception:
    __version__ = 'unknown'

__all__ = ["DataProto", "__version__"]
