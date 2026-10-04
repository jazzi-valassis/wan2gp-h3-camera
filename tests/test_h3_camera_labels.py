import ast
from pathlib import Path
import unittest

import gradio as gr
from shared.utils import plugins as host_plugins

from plugin_under_test import load_plugin

camera = load_plugin()
WGP = Path(host_plugins.__file__).resolve().parents[2] / "wgp.py"


def load_host_functions(names, namespace):
    tree = ast.parse(WGP.read_text(encoding="utf-8"), filename=str(WGP))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    found = {node.name for node in functions}
    missing = set(names) - found
    if missing:
        raise AssertionError(f"Missing host functions: {sorted(missing)}")
    code = compile(ast.Module(body=functions, type_ignores=[]), str(WGP), "exec")
    namespace = dict(namespace)
    exec(code, namespace)
    return namespace


class NativeLabelHelperTests(unittest.TestCase):
    def test_native_prompt_refresh_reports_fg_after_pg_and_window_modes(self):
        models = {"h3": {"architecture": "minimax_h3"}}
        host = load_host_functions(
            {"get_state_model_type", "get_model_def", "get_prompt_labels", "get_image_end_label", "refresh_prompt_labels"},
            {"gr": gr, "models_def": models},
        )
        host["render_prompt_info_label"] = lambda label, model, definition, prompt_id: f"{prompt_id}:{label}"
        state = {"model_type": "h3"}
        fg = host["refresh_prompt_labels"](state, "FG", 0)
        self.assertIn("all the Lines are Parts of the Same Prompt", fg[0]["label"])
        self.assertIn("all the Lines are Parts of the Same Prompt", fg[1]["label"])
        self.assertIn(fg[0]["label"], fg[3]["value"])
        self.assertIn(fg[1]["label"], fg[4]["value"])
        for previous_mode in ("PG", "W"):
            previous = host["refresh_prompt_labels"](state, previous_mode, 0)
            self.assertNotEqual(previous[0]["label"], fg[0]["label"])
            self.assertNotIn("all the Lines are Parts of the Same Prompt", previous[0]["label"])

    def test_native_frame_label_uses_normalized_frame_count(self):
        models = {"h3": {"architecture": "minimax_h3"}, "minimax_h3": {"fps": 24}}
        host = load_host_functions(
            {"is_integer", "get_state_model_type", "get_model_def", "get_base_model_type", "get_model_fps",
             "get_computed_fps", "compute_video_length_label", "refresh_video_length_label"},
            {"gr": gr, "models_def": models, "model_types_handlers": {"minimax_h3": object()}},
        )
        normalized = camera.normalize_frame_count(225, 107, 17, 5)
        self.assertEqual(normalized, 226)
        update = host["refresh_video_length_label"]({"model_type": "h3"}, normalized, "24", None, None)
        self.assertEqual(update["label"], "Number of frames (24 frames = 1s), current duration: 9.4s")


if __name__ == "__main__":
    unittest.main()
