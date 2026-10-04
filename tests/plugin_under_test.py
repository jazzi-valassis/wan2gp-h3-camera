import importlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "h3_camera_under_test"


def load_plugin():
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{PACKAGE}.plugin")
