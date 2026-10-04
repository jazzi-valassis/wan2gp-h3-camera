"""Self-contained, sandboxed camera editor and its Gradio value bridge."""

from html import escape
from pathlib import Path


_WEB = Path(__file__).with_name("web")


def render_editor():
    """Embed local assets; no additional routes, packages or network requests."""
    document = (_WEB / "editor.html").read_text(encoding="utf-8")
    document = document.replace("/* CAMERA_EDITOR_CSS */", (_WEB / "editor.css").read_text(encoding="utf-8"))
    document = document.replace("/* CAMERA_EDITOR_JS */", (_WEB / "editor.js").read_text(encoding="utf-8"))
    return ('<iframe id="h3-camera-editor" title="H3 camera path editor" '
            'sandbox="allow-scripts" style="width:100%;height:530px;border:0;display:block;" '
            'srcdoc="' + escape(document, quote=True) + '"></iframe>')


BRIDGE_JS = (_WEB / "bridge.js").read_text(encoding="utf-8")
