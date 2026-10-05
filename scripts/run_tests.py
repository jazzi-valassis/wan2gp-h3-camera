import argparse
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Test this H3 Camera package against a WanGP installation.")
    parser.add_argument("--host", type=Path, help="WanGP root; inferred when installed under its plugins folder.")
    parser.add_argument("--suite", choices=("all", "plan", "integration", "native"), default="all")
    args = parser.parse_args()
    host = (args.host or ROOT.parent.parent).resolve()
    if args.suite != "plan":
        if not (host / "wgp.py").is_file():
            parser.error("Pass --host pointing to the WanGP installation, or use --suite plan.")
        os.environ["WAN2GP_TEST_HOST"] = str(host)
        sys.path.insert(0, str(host))
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    patterns = {
        "all": ("test_h3_camera*.py",),
        "plan": ("test_h3_camera_plan.py", "test_h3_camera_elevation.py"),
        "integration": ("test_h3_camera_plugin.py", "test_h3_camera_labels.py"),
        "native": ("test_h3_camera_roundtrip.py",),
    }
    suite = unittest.TestSuite()
    for pattern in patterns[args.suite]:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
