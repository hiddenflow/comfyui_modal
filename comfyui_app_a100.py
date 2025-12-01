import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"

# ComfyUI default install location
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd
    
def hf_download(repoid: id, filename: str, subdir: str, subfolder: Optional[str] = None, revision: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL, revision=revision)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        "pip install librosa",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia",
        "pip install misaki[en]",
        "pip install ninja",
        "pip install psutil",
        "pip install packaging",
        "pip install wheel",
        "pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0",
        "pip install http://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("crystian/ComfyUI-Crystools", {'install_reqs': True}),
    ("kijai/ComfyUI-WanVideoWrapper", {'install_reqs': True}),
    ("Gourieff/ComfyUI-ReActor", {'install_reqs': True}),
    ("Kosinkadink/ComfyUI-VideoHelperSuite", {'install_reqs': True})
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Model download tasks (will be done at runtime)
model_tasks = [
    ("city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q4_0.gguf", "diffusion_models", None, None),
    ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras", None, None),
    ("Kijai/WanVideo_comfy_GGUF", "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf", "diffusion_models", None, None),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae", None, None),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision", None, None),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders", None, None),
    ("Kijai/wav2vec2_safetensors", "wav2vec2-chinese-base_fp16.safetensors", "wav2vec2", None, None),
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "clip", None, None),
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-fp8_e4m3fn.safetensors", "clip", None, None),
    ("facefusion/models-3.3.0", "hyperswap_1a_256.onnx", "hyperswap", None, None),
    ("Kijai/Z-Image_comfy_fp8_scaled", "z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors", "diffusion_models", None, None),
    ("jiangchengchengNLP/qwen3-4b-fp8-scaled", "qwen3_4b_fp8_scaled.safetensors", "clip", None, None),
    ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", "vae", None, None),
    ("Kijai/WanVideo_comfy_fp8_scaled", "T2V/HoloCine/Wan2_2-T2V-A14B-LOW-HoloCine-full_fp8_e4m3fn_scaled_KJ.safetensors", "vae", None, None),
    ("Kijai/WanVideo_comfy_fp8_scaled", "T2V/HoloCine/Wan2_2-T2V-A14B-HIGH-HoloCine-full_fp8_e4m3fn_scaled_KJ.safetensors", "vae", None, None),
#    ("Kijai/LongCat-Video_comfy", "LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models", None, None),
#    ("Kijai/LongCat-Video_comfy", "LongCat_distill_lora_alpha64_bf16.safetensors", "loras", None, None),
#    ("Kijai/LongCat-Video_comfy", "LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models", None, None),
    ("lightx2v/Wan2.2-Distill-Models", "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors", "diffusion_models", None, None),
    ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", "loras", None, None),
    ("Kijai/WanVideo_comfy_fp8_scaled", "Fun/Wan2_2-Fun-InP-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models", None, None),
    ("Kijai/WanVideo_comfy_fp8_scaled", "Fun/Wan2_2-Fun-InP-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models", None, None),
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/Phhofm/models/releases/download/1xgaterv3_r_sharpen/1xgaterv3_r_sharpen_fp32_op17_onnxslim.onnx -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/Phhofm/models/releases/download/2xPublic_realplksr_dysample_layernorm_real_nn/2xPublic_realplksr_dysample_layernorm_real_nn.pth -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/Phhofm/models/releases/download/2xParagonSR_Nano_gan/2xParagonSR_Nano_gan_op18_fp16.onnx -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/GFPGANv1.4.onnx -P {MODELS_DIR}/facerestore_models",
    f"wget https://github.com/Glat0s/GFPGAN-1024-onnx/releases/download/v0.0.1/gfpgan-1024.onnx -P {MODELS_DIR}/facerestore_models",
    f"wget https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/RestoreFormerPlusPlus.fp16.onnx -P {MODELS_DIR}/facerestore_models",
    
]

# Create volume
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=600,
    timeout=1800,
    gpu=os.environ.get('MODAL_GPU_TYPE', 'L40S'),
    volumes={DATA_ROOT: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)  # Increased timeout for handling restarts
def ui():
    # Check if volume is empty (first run)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI from default location to volume...")
        
        # Ensure DATA_ROOT exists
        os.makedirs(DATA_ROOT, exist_ok=True)
        
        # Copy ComfyUI from default location to volume
        if os.path.exists(DEFAULT_COMFY_DIR):
            print(f"Copying {DEFAULT_COMFY_DIR} to {DATA_BASE}")
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            print(f"Warning: {DEFAULT_COMFY_DIR} not found, creating empty structure")
            os.makedirs(DATA_BASE, exist_ok=True)
    
    # Fix detached HEAD and update ComfyUI backend to the latest version
    print("Fixing git branch and updating ComfyUI backend to the latest version...")
    os.chdir(DATA_BASE)
    try:
        # Check if in detached HEAD state
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Detected detached HEAD, checking out main branch...")
            subprocess.run("git checkout -B main origin/main", shell=True, check=True, capture_output=True, text=True)
            print("Successfully checked out main branch")
        # Configure pull strategy to fast-forward only
        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        # Perform git pull
        result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        print("Git pull output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI backend: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during backend update: {e}")

    # Update ComfyUI-Manager to the latest version
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Updating ComfyUI-Manager to the latest version...")
        os.chdir(manager_dir)
        try:
            # Configure pull strategy for ComfyUI-Manager
            subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
            result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager git pull output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI-Manager: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during ComfyUI-Manager update: {e}")
        os.chdir(DATA_BASE)  # Return to base directory
    else:
        print("ComfyUI-Manager directory not found, installing...")
        try:
            subprocess.run("comfy node install ComfyUI-Manager", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing ComfyUI-Manager: {e.stderr}")

    # Upgrade pip at runtime
    print("Upgrading pip at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True, check=True, capture_output=True, text=True)
        print("pip upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during pip upgrade: {e}")

    # Upgrade comfy-cli at runtime
    print("Upgrading comfy-cli at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=True, capture_output=True, text=True)
        print("comfy-cli upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading comfy-cli: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during comfy-cli upgrade: {e}")

    # Update ComfyUI frontend by installing requirements
    print("Updating ComfyUI frontend by installing requirements...")
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            result = subprocess.run(
                f"/usr/local/bin/python -m pip install -r {requirements_path}",
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print("Frontend update output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI frontend: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during frontend update: {e}")
    else:
        print(f"Warning: {requirements_path} not found, skipping frontend update")

    # Configure ComfyUI-Manager: Disable auto-fetch, set weak security, and disable file logging
    manager_config_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    print("Configuring ComfyUI-Manager: Disabling auto-fetch, setting security_level to weak, and disabling file logging...")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)
    print(f"Updated {manager_config_path} with network_mode=private, security_level=weak, log_to_file=false")

    # Ensure all required directories exist
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # Download models at runtime (only if missing)
    print("Checking and downloading missing models...")
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            print(f"Downloading {fn} to {target}...")
            try:
                hf_download(sub, fn, repo, subf)
                print(f"Successfully downloaded {fn}")
            except Exception as e:
                print(f"Error downloading {fn}: {e}")
        else:
            print(f"Model {fn} already exists, skipping download")

    # Run extra download commands
    print("Running additional downloads...")
    for cmd in extra_cmds:
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Command completed successfully")
            else:
                print(f"Command failed with return code {result.returncode}: {result.stderr}")
        except Exception as e:
            print(f"Error running command {cmd}: {e}")

    # Set COMFY_DIR environment variable to volume location
    os.environ["COMFY_DIR"] = DATA_BASE
    
    # Launch ComfyUI from volume location
    print(f"Starting ComfyUI from {DATA_BASE}...")
    
    # Start ComfyUI server with correct syntax and latest frontend
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"]
    print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
