import mimetypes
import os
import signal
import sys
import tempfile
import time
from collections import namedtuple
from pathlib import Path

import cmd_args
import gradio as gr
import gradio.routes
import gradio.utils
import launch
import logger
import paths
import settings
import tab_main
import tab_settings
import utilities
from dte_instance import dte_instance
from PIL import Image, PngImagePlugin
from shared_state import state

# ================================================================
# brought from AUTOMATIC1111/stable-diffusion-webui and modified


mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

Savedfile = namedtuple("Savedfile", ["name"])
GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
git = "git"
stored_commit_hash = None
interface = None


def cleanup_tmpdr():
    if not state.temp_dir or not state.temp_dir.is_dir():
        return

    for p in state.temp_dir.glob("**/*.png"):
        if p.is_file():
            os.remove(p)


def register_tmp_file(gradio: gr.Blocks, filename):
    if hasattr(gradio, "temp_file_sets"):  # gradio >=3.15
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {
            os.path.abspath(filename)
        }


def save_pil_to_cache(pil_image: Image.Image, *args, **kwargs):
    already_saved_as = getattr(pil_image, "already_saved_as", None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(interface, already_saved_as)
        return str(Path(already_saved_as).resolve())

    tmpdir = state.temp_dir
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    if tmpdir:
        if not tmpdir.is_dir():
            tmpdir.mkdir(parents=True)
        file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=tmpdir)
    else:
        file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))

    pil_image.already_saved_as = file_obj.name

    return file_obj.name


def save_file_to_cache_nocache(file_path: str | Path, cache_dir: str) -> str:
    return str(Path(file_path).resolve())


def save_file_to_cache_cacheonce(file_path: str | Path, cache_dir: str) -> str:
    """Returns a temporary file path for a copy of the given file path if it does
    not already exist. Otherwise returns the path to the existing temp file."""
    import hashlib

    filename = hashlib.md5(file_path.encode()).hexdigest()
    temp_dir = Path(cache_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    import shutil

    from gradio_client import utils as client_utils

    filename = client_utils.strip_invalid_filename_characters(filename)
    full_temp_file_path = str(Path(temp_dir / filename).resolve())

    if not Path(full_temp_file_path).exists():
        shutil.copy2(file_path, full_temp_file_path)

    return full_temp_file_path


def webpath(fn: Path):
    path = str(fn.absolute()).replace("\\", "/")
    return f"file={path}?{os.path.getmtime(fn)}"


def reload_javascript():

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        command = f'cd "{utilities.base_dir()}" & {git} rev-parse HEAD'
        result = launch.run(command)
        stored_commit_hash = result.stdout.decode("utf-8").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def versions_html():
    # import torch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = commit_hash()
    short_commit = commit[0:8]

    return f"""
python: <span title="{sys.version}">{python_version}</span>
 ･
gradio: {gr.__version__}
 ･
commit: <a href="https://github.com/toshiaki1729/dataset-tag-editor-standalone/commit/{commit}">{short_commit}</a>
"""


def create_ui():
    reload_javascript()

    with gr.Blocks(analytics_enabled=False, title="Dataset Tag Editor") as gui:
        with gr.Tab("Main"):
            tab_main.on_ui_tabs()
        with gr.Tab("Settings"):
            tab_settings.on_ui_tabs()

        gr.Textbox(elem_id="ui_created", value="", visible=False)

        footer = f'<div class="versions">{versions_html()}</div>'
        gr.HTML(footer, elem_id="footer")
    return gui


def wait_on_server():
    global interface
    while True:
        time.sleep(0.5)
        if state.need_restart:
            state.need_restart = False
            if interface:
                time.sleep(0.25)
                interface.close()
                time.sleep(0.25)
            break


# ================================================================


def main():
    global interface

    def sigint_handler(sig, frame):
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    while True:
        state.begin()

        settings.load()
        paths.paths = paths.Paths()

        if settings.current.use_temp_files:
            state.temp_dir = Path(
                settings.current.temp_directory or utilities.base_dir_path() / "temp"
            ).absolute()
            os.environ["GRADIO_TEMP_DIR"] = str(state.temp_dir)
        else:
            state.temp_dir = None
            # PILのキャッシュを元のGradioの実装に戻す
            gr.gradio.processing_utils.save_pil_to_cache = (
                gr.processing_utils.save_pil_to_cache
            )

        # ファイルキャッシュの設定は条件に関係なく行われる
        gr.gradio.processing_utils.save_file_to_cache = save_file_to_cache_nocache

        if settings.current.cleanup_tmpdir:
            cleanup_tmpdr()

        dte_instance.load_interrogators()

        interface = create_ui().queue(64)

        # Always allow project root directory
        project_root = utilities.base_dir_path()
        allowed_paths = [str(project_root)]
        if settings.current.allowed_paths:
            allowed_paths.extend(
                [
                    str(Path(path).absolute())
                    for path in settings.current.allowed_paths.split(", ")
                ]
            )

        app, _, _ = interface.launch(
            server_port=cmd_args.opts.port,
            server_name=cmd_args.opts.server_name,
            share=cmd_args.opts.share,
            auth=[tuple(cred.split(":")) for cred in cmd_args.opts.auth]
            if cmd_args.opts.auth
            else None,
            ssl_keyfile=cmd_args.opts.tls_key,
            ssl_certfile=cmd_args.opts.tls_cert,
            debug=cmd_args.opts.gradio_debug,
            prevent_thread_lock=True,
            allowed_paths=allowed_paths,
        )

        # Disable a very open middleware as Stable Diffusion web UI does
        app.user_middleware = [
            x for x in app.user_middleware if x.cls.__name__ != "CORSMiddleware"
        ]

        wait_on_server()
        logger.write("Restarting UI...")
