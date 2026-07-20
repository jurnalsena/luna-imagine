import os
import re
import unicodedata
from pathlib import Path
import glob
import time
import json
import base64
import subprocess
import requests
import warnings
import gradio as gr


from requests.exceptions import ReadTimeout

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*browser-compatible container.*")

SERVER_URL = "http://127.0.0.1:1234"
if os.path.exists("/teamspace/studios/this_studio"):
    LOG_PATH = "/teamspace/studios/this_studio/server.log"
else:
    LOG_PATH = "/kaggle/working/server.log"

def get_working_dir():
    """Returns the environment-specific directory for generated outputs."""
    if os.path.exists("/teamspace/studios/this_studio"):
        return "/teamspace/studios/this_studio/outputs"
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working"
    return "/tmp/free-aistudio"

def get_models_base():
    """Returns the base models directory depending on the environment."""
    if os.path.exists("/teamspace/studios/this_studio"):
        return "/teamspace/studios/this_studio/models"
    return "/tmp/models"

def is_lightning_studio():
    return os.path.exists("/teamspace/studios/this_studio")

def get_upscaler_info():
    """Scans the latent_upscale_models directory for a safetensors upscaler."""
    base = get_models_base()
    upscale_dir = os.path.join(base, "latent_upscale_models")
    if os.path.exists(upscale_dir):
        import glob
        files = glob.glob(os.path.join(upscale_dir, "*.safetensors"))
        if files:
            path = files[0]
            name = os.path.splitext(os.path.basename(path))[0]
            return path, name
    # Fallback to standard Kaggle/Default path
    fallback = os.path.join(base, "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    return fallback, os.path.splitext(os.path.basename(fallback))[0]

def get_vae_tiling_params(enable_tiling):
    if not enable_tiling:
        return {"enabled": False}
    return {
        "enabled": True,
        "temporal_tiling": True,
        "tile_size_x": 16,
        "tile_size_y": 16,
        "target_overlap": 0.25,
        "rel_size_x": 0.0,
        "rel_size_y": 0.0,
        "extra_tiling_args": "temporal_tile_frames=4,temporal_tile_overlap=1",
    }

def get_live_logs():
    """Reads the tail end of the server log file to stream into the interface."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
            return "".join(lines[-20:])
    return "Waiting for server logs to initialize..."

def scan_history():
    """Scans the working directory for generated video outputs."""
    working_dir = get_working_dir()
    video_files = (
        glob.glob(os.path.join(working_dir, "luna_*.webm"))
        + glob.glob(os.path.join(working_dir, "luna_*.avi"))
        + glob.glob(os.path.join(working_dir, "luna_*.mp4"))
    )
    video_files.sort(key=os.path.getmtime, reverse=True)
    return video_files

def make_preview_video(video_path):
    """Returns a browser-friendly MP4 preview while preserving the original output."""
    import shutil
    if not video_path.lower().endswith(".avi"):
        return video_path

    preview_path = os.path.splitext(video_path)[0] + ".mp4"
    
    # Locate ffmpeg executable
    ffmpeg_cmd = "ffmpeg"
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"🎬 Found imageio-ffmpeg static binary: {ffmpeg_cmd}")
        except ImportError:
            print("⚠️ Warning: ffmpeg not found in PATH and imageio-ffmpeg is not installed. Video container conversion might fail.")
            
    print(f"🎬 Starting ffmpeg conversion using: {ffmpeg_cmd}")
    try:
        # 1. Attempt full conversion including audio stream encoding
        res = subprocess.run(
            [
                ffmpeg_cmd,
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                preview_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
            print("✅ Successfully encoded video preview with audio.")
            return preview_path
    except Exception as e:
        print(f"⚠️ Pass 1 ffmpeg conversion failed. Error: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"ffmpeg Pass 1 Stderr:\n{e.stderr}")
            
        # 2. Fallback to video-only conversion if the file has no audio stream or audio codec fails
        try:
            print("🎬 Retrying ffmpeg conversion without audio (-an)...")
            res2 = subprocess.run(
                [
                    ffmpeg_cmd,
                    "-y",
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    "-movflags",
                    "+faststart",
                    preview_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
                print("✅ Successfully encoded video preview (video-only).")
                return preview_path
        except Exception as e2:
            print(f"⚠️ Pass 2 ffmpeg conversion failed. Error: {e2}")
            if hasattr(e2, 'stderr') and e2.stderr:
                print(f"ffmpeg Pass 2 Stderr:\n{e2.stderr}")

    return video_path

def build_failure_message(status_res):
    """Turns the server job response and recent logs into a beginner-readable UI message."""
    parts = []
    for key in ("error", "message", "detail"):
        if status_res.get(key):
            parts.append(str(status_res[key]))

    result = status_res.get("result")
    if isinstance(result, dict):
        for key in ("error", "message", "detail"):
            if result.get(key):
                parts.append(str(result[key]))

    recent_logs = get_live_logs()
    important_logs = [
        line for line in recent_logs.splitlines()
        if "[ERROR]" in line or "[WARN" in line or "requires hires upscaler" in line.lower()
    ]
    if important_logs:
        parts.append("Recent engine log:\n" + "\n".join(important_logs[-6:]))
        lower_logs = "\n".join(important_logs).lower()
        if "out of memory" in lower_logs or "failed to allocate" in lower_logs:
            parts.append("What happened: the video was generated and upscaled, but final VAE decoding needed more GPU memory than available. Try the 360p preset, fewer frames, or keep the smaller upscale VAE tile settings in this notebook.")

    if not parts:
        parts.append(json.dumps(status_res, indent=2)[:2000])

    return "Video generation failed.\n\n" + "\n\n".join(parts)

# =====================================================================
# Prompt guardrail (inline — works when ui.py is imported as src.ui)
# =====================================================================

try:
    from prompt_guardrail import enforce_prompt_safety
except ImportError:
    try:
        from .prompt_guardrail import enforce_prompt_safety
    except ImportError:
        _BLOCKED_TERMS = {
            "porn", "porno", "pornography", "pornographic", "xxx", "nsfw", "hentai", "ecchi",
            "bokep", "jav", "xvideo", "xvideos", "redtube", "youporn", "onlyfans",
            "nude", "nudes", "nudity", "naked", "topless", "bottomless", "bare breasts",
            "full frontal", "exposed breasts", "exposed genitals", "no clothes", "without clothes",
            "telanjang", "bugil", "payudara", "vagina", "penis", "testicle", "testicles",
            "genital", "genitals", "areola", "nipple", "nipples", "labia", "clitoris",
            "sex", "sexual", "intercourse", "blowjob", "handjob", "fellatio", "cunnilingus",
            "anal sex", "oral sex", "gangbang", "threesome", "orgy", "masturbat", "masturbation",
            "ejaculat", "cumshot", "creampie", "deepthroat", "doggy style", "missionary position",
            "sodomy", "bestiality", "zoophilia", "rape", "raping", "molest", "molestation",
            "pemesek", "ngewe", "ngentot", "bersetubuh", "seks", "seksual",
            "fetish", "bdsm", "bondage", "dominatrix", "stripper", "strip club", "lap dance",
            "upskirt", "downblouse", "cameltoe", "underboob", "sideboob", "wardrobe malfunction",
            "see through", "see-through", "sheer lingerie", "micro bikini", "g-string",
            "loli", "lolicon", "shota", "shotacon", "child porn", "cp ", "underage",
            "minor nude", "teen nude", "schoolgirl nude",
        }
        _BLOCKED_PATTERNS = [
            r"\bnsfw\b",
            r"\bxxx+\b",
            r"\b(?:p0rn|pr0n|nud3|n@ked|s3x|s€x)\b",
            r"\b(?:no|without|without any|w\/o)\s+(?:clothes|clothing|outfit|panties|bra|underwear)\b",
            r"\b(?:remove|removing|take off|taking off)\s+(?:clothes|clothing|bra|panties|underwear|top|shirt)\b",
            r"\b(?:spread(?:ing)?|open(?:ing)?)\s+(?:legs|thighs)\b",
            r"\b(?:touch(?:ing)?|grab(?:bing)?|grop(?:ing|e))\s+(?:breasts?|boobs?|butt|genitals?)\b",
            r"\b(?:nude|naked|nsfw)\s+(?:girl|woman|women|boy|man|teen|model|photo|pic|image|selfie)\b",
            r"\b(?:erotic|sensual|provocative)\s+(?:photo|pose|picture|scene|art)\b.*\b(?:nude|naked|explicit)\b",
            r"\b(?:explicit|graphic)\s+(?:sex|sexual|nude|adult)\b",
            r"\b(?:adult|18\+)\s+(?:content|video|scene|film)\b",
            r"\b(?:onlyfans|playboy|penthouse)\s+(?:style|model|photo)\b",
            r"\b(?:bokep|film\s*dewasa|video\s*dewasa)\b",
            r"\b(?:telanjang|bugil)\s+(?:wanita|perempuan|cewek|gadis|pria|cowok)\b",
        ]
        _LEET_MAP = str.maketrans({
            "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
            "@": "a", "$": "s", "!": "i",
        })
        _GUARDRAIL_MESSAGE = (
            "This tool does not allow pornographic, sexually explicit, or NSFW requests.\n"
            "Please revise your prompt and try again."
        )

        def _normalize_prompt(text: str) -> str:
            if not text:
                return ""
            text = unicodedata.normalize("NFKC", str(text)).lower()
            text = text.translate(_LEET_MAP)
            text = re.sub(r"[_\-.]+", " ", text)
            text = re.sub(r"[^\w\s]", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        def _find_blocked_term(text: str):
            padded = f" {text} "
            for term in sorted(_BLOCKED_TERMS, key=len, reverse=True):
                if f" {term} " in padded:
                    return term
                if term.isalpha() and re.search(rf"\b{re.escape(term)}\b", text):
                    return term
            return None

        def enforce_prompt_safety(prompt: str, *, field_name: str = "Prompt") -> None:
            normalized = _normalize_prompt(prompt)
            if not normalized:
                return
            blocked_term = _find_blocked_term(normalized)
            if blocked_term:
                raise ValueError(
                    f"{field_name} blocked by content guardrail.\n\n"
                    f"Blocked term detected: '{blocked_term}'\n\n{_GUARDRAIL_MESSAGE}"
                ) from None
            for pattern in _BLOCKED_PATTERNS:
                if re.search(pattern, normalized, flags=re.IGNORECASE):
                    raise ValueError(
                        f"{field_name} blocked by content guardrail.\n\n"
                        f"Blocked phrase pattern detected.\n\n{_GUARDRAIL_MESSAGE}"
                    ) from None

def _guard_prompt(prompt: str, field_name: str = "Prompt") -> None:
    """Raises gr.Error when the prompt fails the NSFW guardrail."""
    try:
        enforce_prompt_safety(prompt, field_name=field_name)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

def handle_generation(prompt, negative_prompt, steps, resolution_preset, use_custom_resolution, custom_width, custom_height, duration_seconds, input_image, enable_upscale, cfg_scale, distilled_guidance, scheduler, flow_shift, enable_vae_tiling, enable_guardrail, enable_image_upload):
    """Processes frontend inputs and generates video using either CLI (Lightning.ai) or HTTP API (Kaggle)."""
    if enable_guardrail:
        _guard_prompt(prompt, "Text prompt")
        _guard_prompt(negative_prompt, "Negative prompt")
    
    # If image upload is disabled, ignore the input image
    if not enable_image_upload:
        input_image = None

    if use_custom_resolution:
        width, height = int(custom_width), int(custom_height)
        if width % 32 != 0 or height % 32 != 0:
            raise gr.Error("Custom width and height must both be divisible by 32.")
        if width < 256 or height < 256:
            raise gr.Error("Custom width and height must be at least 256 pixels.")
        if width > 1920 or height > 1088:
            raise gr.Error("Custom resolution is capped at 1920x1088.")
    elif "360p" in resolution_preset:
        width, height = 480, 360  # Proven fast baseline
    elif "480p" in resolution_preset:
        width, height = 640, 368  # Proven balanced size
    else:
        width, height = 832, 480

    fps = 24 if is_lightning_studio() else 12  # Use 24 fps on Lightning as per user CLI test
    target_frames = max(9, int(round(float(duration_seconds) * fps)))
    frames = min(121, ((target_frames - 1 + 8) // 8) * 8 + 1)  # LTX video frame count rule: 8N + 1.

    if is_lightning_studio():
        # Setup paths
        bin_dir = "/teamspace/studios/this_studio/sd_bin"
        cli_path = os.path.join(bin_dir, "bin/sd-cli")
        models_base = "/teamspace/studios/this_studio/models"
        working_dir = get_working_dir()
        os.makedirs(working_dir, exist_ok=True)
        
        job_id = str(int(time.time()))
        output_ext = "webm"
        base_video_path = os.path.join(working_dir, f"luna-imagine_{job_id}.{output_ext}")
        
        # Build command-line list
        cmd = [
            cli_path,
            "-M", "vid_gen",
            "--diffusion-model", os.path.join(models_base, "diffusion_models/ltx-2.3-22b-distilled-Q8_0.gguf"),
            "--vae", os.path.join(models_base, "vae/ltx-2.3-22b-distilled_video_vae.safetensors"),
            "--llm", os.path.join(models_base, "text_encoders/gemma-3-12b-it-Q6_K.gguf"),
            "--embeddings-connectors", os.path.join(models_base, "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"),
            "-p", str(prompt),
            "-n", str(negative_prompt),
            "--cfg-scale", str(cfg_scale),
            "--guidance", str(distilled_guidance),
            "--sampling-method", "euler",
            "--steps", str(steps),
            "-W", str(width),
            "-H", str(height),
            "--video-frames", str(frames),
            "--fps", str(fps),
            "-o", base_video_path,
            "-v"
        ]
        
        if os.path.exists(os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")):
            cmd += ["--audio-vae", os.path.join(models_base, "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")]
            
        cmd += ["--offload-to-cpu"]
        cmd += ["--diffusion-fa"]
        
        if scheduler != "default" and scheduler != "none":
            cmd += ["--scheduler", str(scheduler)]
            if flow_shift > 0:
                cmd += ["--flow-shift", str(flow_shift)]
        elif flow_shift > 0 and abs(flow_shift - 2.37) > 0.001 and abs(flow_shift - 1.3568) > 0.001:
            cmd += ["--flow-shift", str(flow_shift)]

        if enable_vae_tiling:
            cmd += ["--vae-tiling"]
            cmd += ["--extra-tiling-args", "temporal_tile_frames=4,temporal_tile_overlap=1"]

        if input_image is not None and os.path.exists(input_image):
            cmd += ["--init-image", str(input_image)]
            
        print(f"🚀 Running CLI generation command:\n{' '.join(cmd)}")
        
        env = os.environ.copy()
        from src.server import find_cuda_library_paths
        valid_paths = find_cuda_library_paths()
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        if existing_ld:
            env["LD_LIBRARY_PATH"] = ":".join(valid_paths) + ":" + existing_ld
        else:
            env["LD_LIBRARY_PATH"] = ":".join(valid_paths)
            
        # Write real-time output to LOG_PATH
        log_file = open(LOG_PATH, "w")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
            
            while process.poll() is None:
                time.sleep(1)
                
            log_file.close()
            
            if process.returncode != 0:
                with open(LOG_PATH, "r") as f:
                    recent_logs = "".join(f.readlines()[-30:])
                raise gr.Error(f"CLI generation failed with code {process.returncode}.\n\nRecent logs:\n{recent_logs}")
                
            print(f"✅ CLI generation completed! Saved to {base_video_path}")
            return make_preview_video(base_video_path)
            
        except Exception as e:
            try:
                log_file.close()
            except Exception:
                pass
            raise gr.Error(f"Error during CLI execution: {e}")

    else:
        # HTTP API generation (Kaggle or local fallback)
        payload = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt),
            "width": int(width),
            "height": int(height),
            "strength": 0.75 if input_image else 1.0,
            "seed": -1,
            "video_frames": int(frames),
            "fps": fps,
            "moe_boundary": 0.875,
            "vace_strength": 1.0,
            "sample_params": {
                "scheduler": str(scheduler) if scheduler != "default" else "discrete",
                "sample_method": "euler",
                "sample_steps": int(steps),
                "flow_shift": float(flow_shift) if flow_shift > 0 else (1.3568 if scheduler == "discrete" or scheduler == "default" else 2.37),
                "guidance": {
                    "txt_cfg": float(cfg_scale),
                    "img_cfg": float(cfg_scale),
                    "distilled_guidance": float(distilled_guidance)
                },
            },
            "vae_tiling_params": get_vae_tiling_params(enable_vae_tiling),
            "output_format": "avi",
            "output_compression": 100,
        }

        if "audio" not in payload["prompt"].lower():
            payload["prompt"] = f"{payload['prompt']}, high quality clear audio"

        if enable_upscale:
            upscaler_path, upscaler_name = get_upscaler_info()
            if not os.path.exists(upscaler_path):
                raise gr.Error(f"Upscaling is enabled, but the upscaler model is missing:\n{upscaler_path}\n\nRun download step first.")

            payload["hires"] = {
                "enabled": True,
                "upscaler": upscaler_name,
                "scale": 2.0,
                "steps": 10,
                "denoising_strength": 0.7,
            }

        if input_image is not None and os.path.exists(input_image):
            with open(input_image, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            image_payload = f"data:image/png;base64,{img_base64}"
            payload["init_image"] = image_payload
            payload["input_image"] = image_payload

        try:
            r = requests.post(f"{SERVER_URL}/sdcpp/v1/vid_gen", json=payload, timeout=30)
            r.raise_for_status()
            job_id = r.json()["id"]
            status_timeouts = 0

            while True:
                try:
                    status_res = requests.get(f"{SERVER_URL}/sdcpp/v1/jobs/{job_id}", timeout=120).json()
                    status_timeouts = 0
                except ReadTimeout:
                    status_timeouts += 1
                    if status_timeouts >= 3:
                        raise gr.Error(
                            "The generation server is not responding to status checks.\n\n"
                            "This usually means the backend is stuck in a long CUDA operation or hit a CUDA error.\n\n"
                            f"Recent logs:\n{get_live_logs()}"
                        )
                    time.sleep(8)
                    continue

                status = status_res.get("status", "unknown")

                if status == "completed":
                    video_bytes = base64.b64decode(status_res["result"]["b64_json"])
                    working_dir = get_working_dir()
                    os.makedirs(working_dir, exist_ok=True)
                    output_ext = payload["output_format"]
                    base_video_path = os.path.join(working_dir, f"luna-imagine_{job_id}.{output_ext}")
                    with open(base_video_path, "wb") as f:
                        f.write(video_bytes)
                    return make_preview_video(base_video_path)

                if status in ("failed", "cancelled"):
                    raise gr.Error(build_failure_message(status_res))

                time.sleep(4)

        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Could not communicate with the generation server.\n\n{type(e).__name__}: {e}\n\nRecent logs:\n{get_live_logs()}")

_UI_DIR = Path(__file__).resolve().parent

def _load_ui_css():
    css_path = _UI_DIR / "ui_theme.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return "footer { display: none !important; }"

def _modern_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0c0c12",
        body_background_fill_dark="#0c0c12",
        block_background_fill="#16161f",
        block_background_fill_dark="#16161f",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_width="1px",
        block_label_text_weight="600",
        block_title_text_weight="700",
        block_radius="14px",
        button_large_radius="12px",
        button_primary_background_fill="linear-gradient(90deg, #6C4DFF 0%, #9B5DE5 50%, #FF4FBF 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #7D5FFF 0%, #AB6DFF 50%, #FF6FCF 100%)",
        button_primary_text_color="white",
        button_secondary_background_fill="#252532",
        button_secondary_background_fill_hover="#2f2f40",
        input_background_fill="#12121a",
        input_border_color="rgba(255,255,255,0.1)",
        shadow_drop="0 8px 32px rgba(0,0,0,0.35)",
    )

def build_app():
    """Constructs and returns the Gradio app blocks."""
    default_resolution = "360p (480x360) - Fastest Testing Baseline" if is_lightning_studio() else "480p (640x368) - Optimized Safe Balanced Size"
    default_duration = 2 if is_lightning_studio() else 5
    default_steps = 6 if is_lightning_studio() else 8

    with gr.Blocks(theme=_modern_theme(), css=_load_ui_css(), elem_id="ltx-app", title="LTX Video Studio") as app:
        gr.HTML(
            """
            <div class="luna-hero">
                <span class="luna-badge">LTX 2.3</span>
                <h1>LTX Video Studio</h1>
                <p>Generate videos with fine-tuned controls. Monitor engine logs and browse your generation history.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="luna-panel"):
                    gr.Markdown("**Controls**", elem_classes="luna-panel-title")
                    prompt = gr.Textbox(label="Text Prompt", placeholder="Describe the video actions and sounds clearly...", lines=3)
                    neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, worst quality, low quality, glitch, distortion", lines=2)

                resolution_preset = gr.Dropdown(
                    choices=[
                        "360p (480x360) - Fastest Testing Baseline",
                        "480p (640x368) - Optimized Safe Balanced Size",
                        "720p (832x480) - High Resolution Cinematic Layout",
                    ],
                    value=default_resolution,
                    label="Core Video Generation Dimensions",
                )

                use_custom_resolution = gr.Checkbox(label="Use Custom Resolution", value=False)
                with gr.Row():
                    custom_width = gr.Slider(minimum=256, maximum=1920, value=640, step=32, label="Custom Width")
                    custom_height = gr.Slider(minimum=256, maximum=1088, value=384, step=32, label="Custom Height")

                duration_seconds = gr.Slider(minimum=1, maximum=10, value=default_duration, step=0.5, label="Duration Seconds (rounded to valid LTX frame count)")
                steps = gr.Slider(minimum=4, maximum=30, value=default_steps, step=1, label="Sampling Steps (LTX 2.3 Distilled Sweet Spot: 8-12)")

                with gr.Accordion("Advanced Generation Settings (Fine-Tuning)", open=False):
                    cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=6.0, step=0.1, label="CFG Scale (txt_cfg / img_cfg)")
                    distilled_guidance = gr.Slider(minimum=1.0, maximum=10.0, value=3.5, step=0.1, label="Distilled Guidance Scale")
                    scheduler = gr.Dropdown(choices=["default", "discrete", "ltx2"], value="default", label="Inference Scheduler")
                    flow_shift = gr.Slider(minimum=0.0, maximum=5.0, value=0.0, step=0.01, label="Flow Shift Parameter (0.0 = Auto-calculate based on scheduler)")
                    enable_vae_tiling = gr.Checkbox(label="Enable VAE Tiling (Disable for A100/A10G to get maximum quality without seams)", value=False if is_lightning_studio() else True)

                enable_upscale = gr.Checkbox(label="Enable Native Hi-Res Upscaling Pass", value=False)
                enable_guardrail = gr.Checkbox(label="Enable Content Guardrail (NSFW filter)", value=False)
                enable_image_upload = gr.Checkbox(label="Enable Image Upload (Image-to-Video)", value=True)
                input_image = gr.Image(label="Input Image (For Image-to-Video)", type="filepath", visible=False)
                generate_btn = gr.Button("Generate New Video", variant="primary", elem_id="ltx-generate-btn", size="lg")
                
                # Toggle input_image visibility when enable_image_upload changes
                def toggle_image_upload(enabled):
                    return gr.update(visible=enabled)
                
                enable_image_upload.change(toggle_image_upload, inputs=[enable_image_upload], outputs=[input_image])

            with gr.Column(scale=1):
                with gr.Group(elem_classes="luna-panel"):
                    gr.Markdown("**Preview (Before / After)**", elem_classes="luna-panel-title")
                    with gr.Row():
                        before_preview = gr.Image(label="Before (Input Image)", type="filepath", visible=False)
                        output_video = gr.Video(label="After (Generated Result)", elem_id="ltx-result")
                
                # Show/hide before preview when enable_image_upload or input_image changes
                def toggle_before_preview(enabled, img):
                    return gr.update(visible=enabled and img is not None, value=img)
                
                enable_image_upload.change(toggle_before_preview, inputs=[enable_image_upload, input_image], outputs=[before_preview])
                input_image.change(toggle_before_preview, inputs=[enable_image_upload, input_image], outputs=[before_preview])
                with gr.Tabs():
                    with gr.Tab("Engine Logs"):
                        log_box = gr.Textbox(label="Live Terminal Output", value="", lines=10, interactive=False, show_label=False)
                        log_timer = gr.Timer(value=3.0, active=True)
                        log_timer.tick(fn=get_live_logs, outputs=log_box)

                    with gr.Tab("History"):
                        refresh_history_btn = gr.Button("Refresh History", variant="secondary", elem_classes="luna-secondary-btn")
                        history_gallery = gr.Gallery(label="Saved Videos (View History)", columns=3, preview=True)
                        history_file = gr.File(label="Saved Videos (Download All)", file_count="multiple")

        generate_btn.click(
            fn=handle_generation,
            inputs=[
                prompt, neg_prompt, steps, resolution_preset, use_custom_resolution,
                custom_width, custom_height, duration_seconds, input_image, enable_upscale,
                cfg_scale, distilled_guidance, scheduler, flow_shift, enable_vae_tiling,
                enable_guardrail, enable_image_upload
            ],
            outputs=output_video,
        ).then(fn=scan_history, outputs=[history_gallery, history_file])

        refresh_history_btn.click(fn=scan_history, outputs=[history_gallery, history_file])
        app.load(fn=scan_history, outputs=[history_gallery, history_file])

    app.queue()
    return app

def launch():
    """Convenience function to start UI immediately in Kaggle."""
    app = build_app()
    app.launch(share=True, inline=False, allowed_paths=[get_working_dir()], show_api=False)

# =====================================================================
# Z-Image-Turbo Image Generation UI Components
# =====================================================================

LORA_DIR = "/tmp/models/loras"
RES_PRESETS = [
    # 1:1 Presets
    ("1:1 (256x256)", 256, 256),
    ("1:1 (512x512)", 512, 512),
    ("1:1 (768x768)", 768, 768),
    ("1:1 (1024x1024)", 1024, 1024),
    ("1:1 (1536x1536)", 1536, 1536),
    ("1:1 (2048x2048) - 2K Square", 2048, 2048),
    # 16:9 Presets
    ("16:9 (640x384)", 640, 384),
    ("16:9 (896x512)", 896, 512),
    ("16:9 (1024x576)", 1024, 576),
    ("16:9 (1536x864)", 1536, 864),
    ("16:9 (1920x1088) - 1080p FHD", 1920, 1088),
    ("16:9 (2048x1152) - 2K Widescreen", 2048, 1152),
    # 9:16 Presets
    ("9:16 (384x640)", 384, 640),
    ("9:16 (512x896)", 512, 896),
    ("9:16 (576x1024)", 576, 1024),
    ("9:16 (864x1536)", 864, 1536),
    ("9:16 (1088x1920) - 1080p FHD Portrait", 1088, 1920),
    ("9:16 (1152x2048) - 2K Portrait", 1152, 2048),
    # 4:3 Presets
    ("4:3 (640x480)", 640, 480),
    ("4:3 (768x576)", 768, 576),
    ("4:3 (1024x768)", 1024, 768),
    ("4:3 (1600x1216)", 1600, 1216),
    ("4:3 (2048x1536) - 2K Standard", 2048, 1536),
    # 3:2 Presets
    ("3:2 (768x512)", 768, 512),
    ("3:2 (1536x1024)", 1536, 1024),
    ("3:2 (2048x1376) - 2K Photo", 2048, 1376),
    # 2:3 Presets
    ("2:3 (512x768)", 512, 768),
    ("2:3 (1024x1536)", 1024, 1536),
    ("2:3 (1376x2048) - 2K Portrait Photo", 1376, 2048),
]
SIZE_OPTIONS = sorted({s for _, w, h in RES_PRESETS for s in (w, h)})

def get_lora_list():
    """List available LoRA files in the loras directory."""
    lora_path = Path(LORA_DIR)
    if not lora_path.exists():
        return []
    return [f.name for f in lora_path.glob("*.safetensors")]

def apply_preset(preset_label):
    for name, w, h in RES_PRESETS:
        if name == preset_label:
            return w, h
    return gr.update(), gr.update()

def scan_image_history():
    """Scans the working directory for generated image outputs."""
    working_dir = get_working_dir()
    image_files = glob.glob(os.path.join(working_dir, "luna-imagine_*.png"))
    image_files.sort(key=os.path.getmtime, reverse=True)
    return image_files

def handle_image_generation(prompt, width, height, steps, seed, cfg_scale, selected_loras, lora_strength, enable_guardrail, input_image, enable_image_upload, denoising_strength=0.75):
    """Processes image params and posts to the API server."""
    if enable_guardrail:
        _guard_prompt(prompt, "Prompt")

    # Append LoRA tags to prompt
    final_prompt = prompt
    if selected_loras:
        from pathlib import Path
        for lora in selected_loras:
            lora_name = Path(lora).stem
            final_prompt += f" <lora:{lora_name}:{lora_strength}>"

    # If image upload is disabled, ignore input_image
    if not enable_image_upload:
        input_image = None

    payload = {
        "prompt": str(final_prompt),
        "negative_prompt": "",
        "width": int(width),
        "height": int(height),
        "seed": int(seed) if int(seed) > 0 else -1,
        "sample_params": {
            "scheduler": "discrete",
            "sample_method": "euler",
            "sample_steps": int(steps),
            "guidance": {
                "txt_cfg": float(cfg_scale),
                "img_cfg": float(cfg_scale),
                "distilled_guidance": float(cfg_scale)
            }
        },
        "output_format": "png",
        "output_compression": 100,
    }
    
    # Add img2img support if input_image is provided
    if input_image is not None:
        payload["strength"] = denoising_strength
        with open(input_image, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            payload["init_image"] = f"data:image/png;base64,{img_base64}"
            payload["input_image"] = f"data:image/png;base64,{img_base64}"

    try:
        r = requests.post(f"{SERVER_URL}/sdcpp/v1/img_gen", json=payload, timeout=30)
        r.raise_for_status()
        job_id = r.json()["id"]

        while True:
            status_res = requests.get(f"{SERVER_URL}/sdcpp/v1/jobs/{job_id}", timeout=10).json()
            status = status_res.get("status", "unknown")

            if status == "completed":
                image_bytes = base64.b64decode(status_res["result"]["images"][0]["b64_json"])
                working_dir = get_working_dir()
                os.makedirs(working_dir, exist_ok=True)
                base_image_path = os.path.join(working_dir, f"gen_{job_id}.png")
                with open(base_image_path, "wb") as f:
                    f.write(image_bytes)
                return base_image_path

            if status in ("failed", "cancelled"):
                raise gr.Error(build_failure_message(status_res))

            time.sleep(1.5) # Fast polling for image

    except Exception as e:
        raise gr.Error(f"Could not communicate with the generation server.\n\n{type(e).__name__}: {e}\n\nRecent logs:\n{get_live_logs()}")

def build_image_app():
    """Constructs and returns the Gradio app blocks for Z-Image-Turbo."""
    with gr.Blocks(theme=_modern_theme(), css=_load_ui_css(), elem_id="luna-app", title="Luna AI Imagine") as app:
        gr.HTML(
            """
            <div class="luna-hero">
                <span class="luna-badge">Cyberus Studio</span>
                <h1>Luna AI Imagine</h1>
                <p>High-speed image generation with resolution presets.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="luna-panel"):
                    gr.Markdown("**Prompt**", elem_classes="luna-panel-title")
                    prompt = gr.Textbox(
                        label="Describe your image",
                        show_label=False,
                        value="Create a premium, modern, minimalist vector logo for a tool engine called: Luna AI Imagine, featuring a cute chibi anime mascot inspired by a young businesswoman with short magenta/pink bob hair, large bright blue eyes, fair skin, wearing a black business blazer, white shirt, black skirt, and black shoes, smiling cheerfully with one eye winking while pointing upward with one hand and resting the other on her waist. Place the mascot inside a modern circular emblem with a purple-to-pink gradient ring inspired by a crescent moon, with a simplified visual novel UI window behind her showing a dialogue box, landscape background, and minimal interface controls in a clean flat vector style. Add subtle sparkles, stars, and a crescent moon icon to reinforce the Luna identity. Use bold rounded geometric sans-serif typography with the text: Luna and a large stylized text: AI, with the subtitle: - IMAGINE - below. Use a premium color palette of purple (#6C4DFF), pink (#FF4FBF), dark navy (#2B2145), and white accents with soft neon gradients. The logo should feel like a premium startup brand, inspired by Japanese anime, with a clean flat vector illustration style, modern UI aesthetic, cute yet professional appearance, symmetrical composition, high contrast, suitable for software branding, website, launcher, GitHub, and application icon, ultra-clean SVG-style vector quality, transparent background, 1:1 aspect ratio, no watermark, no mockup, no realistic rendering, no photorealism, no 3D, no blur, no pixelation, no extra characters, no distorted anatomy, no clutter, no complex background, and no unnecessary decorative elements. 8k",
                        lines=4,
                        placeholder="Describe the image you want to create...",
                    )

                with gr.Group(elem_classes="luna-panel"):
                    gr.Markdown("**Settings**", elem_classes="luna-panel-title")
                    with gr.Row():
                        preset = gr.Dropdown([n for n, _, _ in RES_PRESETS], value="1:1 (512x512)", label="Resolution Preset")
                        steps = gr.Slider(1, 50, value=8, step=1, label="Steps")

                    with gr.Row():
                        width = gr.Dropdown(SIZE_OPTIONS, value=512, label="Width")
                        height = gr.Dropdown(SIZE_OPTIONS, value=512, label="Height")

                    with gr.Row():
                        cfg_scale = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG Scale")
                        seed = gr.Number(value=0, label="Seed (0 = random)")

                with gr.Group(elem_classes=["luna-panel", "luna-lora-box"]):
                    gr.Markdown("### LoRA Support")
                    gr.Markdown("*Place `.safetensors` files in `/tmp/models/loras/`*")
                    with gr.Row():
                        lora_list = gr.CheckboxGroup(choices=get_lora_list(), label="Select LoRAs")
                        refresh_btn = gr.Button("Refresh", variant="secondary", size="sm", elem_classes="luna-secondary-btn")
                    lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="LoRA Strength")

                    def refresh_loras():
                        return gr.update(choices=get_lora_list())
                    refresh_btn.click(refresh_loras, outputs=[lora_list])

                enable_guardrail = gr.Checkbox(label="Enable Content Guardrail (NSFW filter)", value=True)
                enable_image_upload = gr.Checkbox(label="Enable Image Upload (Image-to-Image)", value=True)
                input_image = gr.Image(label="Input Image (For Image-to-Image)", type="filepath", visible=False)
                denoising_strength = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Denoising Strength", visible=False)
                generate_btn = gr.Button("Generate Image", variant="primary", elem_id="luna-generate-btn", size="lg")
                
                # Toggle input_image and denoising_strength visibility when enable_image_upload changes
                def toggle_image_upload(enabled):
                    return (gr.update(visible=enabled), gr.update(visible=enabled))
                
                enable_image_upload.change(toggle_image_upload, inputs=[enable_image_upload], outputs=[input_image, denoising_strength])

            with gr.Column(scale=1):
                with gr.Group(elem_classes="luna-panel"):
                    gr.Markdown("**Preview (Before / After)**", elem_classes="luna-panel-title")
                    with gr.Row():
                        before_preview = gr.Image(label="Before (Input Image)", type="filepath", visible=False)
                        img = gr.Image(label="After (Generated Result)", show_label=False, interactive=False, type="filepath", elem_id="luna-result")
                
                # Show/hide before preview when enable_image_upload or input_image changes
                def toggle_before_preview(enabled, img_input):
                    return gr.update(visible=enabled and img_input is not None, value=img_input)
                
                enable_image_upload.change(toggle_before_preview, inputs=[enable_image_upload, input_image], outputs=[before_preview])
                input_image.change(toggle_before_preview, inputs=[enable_image_upload, input_image], outputs=[before_preview])

                with gr.Tabs():
                    with gr.Tab("Engine Logs"):
                        log_box = gr.Textbox(label="Live Terminal Output", value="", lines=10, interactive=False, show_label=False)
                        log_timer = gr.Timer(value=3.0, active=True)
                        log_timer.tick(fn=get_live_logs, outputs=log_box)

                    with gr.Tab("History"):
                        refresh_history_btn = gr.Button("Refresh History", variant="secondary", elem_classes="luna-secondary-btn")
                        history_gallery = gr.Gallery(label="Saved Images (View History", columns=3, preview=True)
                        history_file = gr.File(label="Saved Images (Download All)", file_count="multiple")

        preset.change(apply_preset, inputs=[preset], outputs=[width, height])

        generate_btn.click(
            fn=handle_image_generation,
            inputs=[prompt, width, height, steps, seed, cfg_scale, lora_list, lora_strength, enable_guardrail, input_image, enable_image_upload, denoising_strength],
            outputs=img,
        ).then(fn=scan_image_history, outputs=[history_gallery, history_file])

        refresh_history_btn.click(fn=scan_image_history, outputs=[history_gallery, history_file])
        app.load(fn=scan_image_history, outputs=[history_gallery, history_file])

    app.queue()
    return app

def launch_image():
    """Convenience function to start Z-Image-Turbo UI immediately in Kaggle."""
    app = build_image_app()
    app.launch(share=True, inline=False, allowed_paths=[get_working_dir()], show_api=False)


