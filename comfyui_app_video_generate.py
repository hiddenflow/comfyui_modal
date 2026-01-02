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
    if "ComfyUI-Frame-Interpolation" in node_repo:
        cmd += "install.py"
    return cmd
    
def hf_download(repo_id: str, filename: str, subdir: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "cudnn-devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
#    modal.Image.debian_slim(python_version="3.12")
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "ffmpeg", "pciutils")
    .apt_install("ninja-build", "build-essential", "python3-dev", "cmake", "clang")
    .run_commands([
        "wget http://archive.ubuntu.com/ubuntu/pool/universe/m/mesa/libgl1-mesa-glx_23.0.4-0ubuntu1~22.04.1_amd64.deb",
        "apt-get install ./libgl1-mesa-glx_23.0.4-0ubuntu1~22.04.1_amd64.deb",
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia",
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PATH": "/usr/local/cuda-12.8/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH",
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "EXT_PARALLEL": "8",
        "NVCC_APPEND_FLAGS": "--threads 4",
        "MAX_JOBS": "16",
        "USE_NINJA": "1",
        "CC": "gcc-13",  # Compiler yang lebih baru
        "CXX": "g++-13",
        "USE_SYSTEM_LIBS": "1"
    })
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("crystian/ComfyUI-Crystools", {'install_reqs': True}),
    ("LykosAI/ComfyUI-Inference-Core-Nodes", {}),
    ("akatz-ai/ComfyUI-Basic-Math", {}),
    ("cubiq/ComfyUI_essentials", {'install_reqs': True}),
    ("yolain/ComfyUI-Easy-Use", {'install_reqs': True}),
    ("jamesWalker55/comfyui-various", {}),
    ("543872524/ComfyUI_crdong", {'install_reqs': True}),
    ("kijai/ComfyUI-WanVideoWrapper", {'install_reqs': True}),
    ("kijai/ComfyUI-KJNodes", {'install_reqs': True}),
    ("kijai/ComfyUI-MelBandRoFormer", {'install_reqs': True}),
    ("Long-form-AI-video-generation/ComfyUI_vaceFramepack", {}),
    ("Fannovel16/ComfyUI-Frame-Interpolation", {}),
    ("numz/ComfyUI-SeedVR2_VideoUpscaler", {'install_reqs': True}),
    ("lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast", {'install_reqs': True}),
    ("kijai/ComfyUI-GIMM-VFI", {'install_reqs': True}),
    ("aining2022/ComfyUI_Swwan", {'install_reqs': True}),
    ("LeonQ8/ComfyUI-Dynamic-Lora-Scheduler", {'install_reqs': True}),
    ("whmc76/ComfyUI-AudioSuiteAdvanced", {'install_reqs': True}),
    ("christian-byrne/audio-separation-nodes-comfyui", {'install_reqs': True}),
    ("Kosinkadink/ComfyUI-VideoHelperSuite", {'install_reqs': True}),
    ("Chaoses-Ib/ComfyUI_Ib_CustomNodes", {'install_reqs': True}),
    ("Gourieff/ComfyUI-ReActor", {'install_reqs': True}),
    ("ClownsharkBatwing/RES4LYF", {'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# pip install
image = image.run_commands([
    "pip install faster-whisper",
    "pip install librosa",
    "pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 xformers==0.0.32.post2 triton==3.4.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall",
    "pip install onnxruntime-gpu",
    "pip install -U setuptools wheel",
    "pip install misaki[en]",
    "pip install ninja",
    "pip install psutil",
    "pip install packaging",
    "pip install soxr==0.5.0.post1 --force-reinstall",
    "pip install numpy==1.26.4 --force-reinstall",
    "export CC=gcc++-13",
    "export CXX=g++-13",
    "git clone https://github.com/thu-ml/SageAttention.git && cd SageAttention && git checkout eb615cf6cf4d221338033340ee2de1c37fbdba4a && python setup.py install",
    "pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.2/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl --no-build-isolation",
    # "git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && python setup.py install",
])

# Model download tasks (will be done at runtime)
model_tasks = [
    ("city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q8_0.gguf", "diffusion_models", None),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "wan2.1_i2v_480p_14B_bf16.safetensors", "diffusion_models", "split_files/diffusion_models"),
    ("Kijai/LongCat-Video_comfy", "LongCat-Avatar_comfy_bf16.safetensors", "diffusion_models", "Avatar"),
    ("Kijai/LongCat-Video_comfy", "LongCat-Avatar-single_fp8_e4m3fn_scaled_mixed_KJ.safetensors", "diffusion_models", "Avatar"),
    ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders", None),
    ("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors", "vae", None),
    # ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "wan_2.1_vae.safetensors", "vae", "split_files/vae"),
    ("Kijai/LongCat-Video_comfy", "LongCat_distill_lora_alpha64_bf16.safetensors", "loras", None),
    ("Kijai/WanVideo_comfy", "Wan2_1-InfiniTetalk-Single_fp16.safetensors", "diffusion_models", "InfiniteTalk"),
    ("Kijai/WanVideo_comfy", "Wan_2_1_T2V_14B_480p_rCM_lora_average_rank_83_bf16.safetensors", "loras", "LoRAs/rCM"),
    ("Kijai/WanVideo_comfy", "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors", "loras", "Lightx2v"),
    ("Kijai/WanVideo_comfy", "UniAnimate-Wan2.1-14B-Lora-12000-fp16.safetensors", "loras", None),
    ("Kijai/MelBandRoFormer_comfy", "MelBandRoformer_fp16.safetensors", "diffusion_models", None),
    ("Kijai/wav2vec2_safetensors", "wav2vec2-chinese-base_fp16.safetensors", "wav2vec2", None),
    ("Kijai/WanVideo_comfy", "LongVie2_attn_layers_bf16.safetensors", "diffusion_models", "LongVie2"),
    ("Kijai/WanVideo_comfy", "longvie2_attn_layers_lora_rank_64_bf16.safetensors", "loras", "LongVie2"),
    ("Kijai/WanVideo_comfy", "LongVie2_dual_controller_controlnet_bf16.safetensors", "controlnet", "LongVie2"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "clip_vision_h.safetensors", "clip_vision", "split_files/clip_vision"),
    ("lightx2v/Wan2.2-Distill-Models", "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors", "diffusion_models", None),
    ("Kijai/WanVideo_comfy", "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors", "loras", "Lightx2v"),
    ("alibaba-pai/Wan2.2-Fun-Reward-LoRAs", "Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors", "loras", None),
    ("alibaba-pai/Wan2.2-Fun-Reward-LoRAs", "Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors", "loras", None),
    ("numz/SeedVR2_comfyUI", "ema_vae_fp16.safetensors", "SEEDVR2", None),
    ("numz/SeedVR2_comfyUI", "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "SEEDVR2", None),
    ("numz/SeedVR2_comfyUI", "seedvr2_ema_7b_fp8_e4m3fn.safetensors", "SEEDVR2", None),
    ("numz/SeedVR2_comfyUI", "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors", "SEEDVR2", None),
    ("JunhaoZhuang/FlashVSR-v1.1", "LQ_proj_in.ckpt", "FlashVSR", None),
    ("JunhaoZhuang/FlashVSR-v1.1", "TCDecoder.ckpt", "FlashVSR", None),
    ("JunhaoZhuang/FlashVSR-v1.1", "Wan2.1_VAE.pth", "FlashVSR", None),
    ("JunhaoZhuang/FlashVSR-v1.1", "diffusion_pytorch_model_streaming_dmd.safetensors", "FlashVSR", None),
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/Phhofm/models/releases/download/4xBHI_dat2_real/4xBHI_dat2_real.pth -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/Phhofm/models/releases/download/1xgaterv3_r_sharpen/1xgaterv3_r_sharpen_fp32_op17_onnxslim.onnx -P {MODELS_DIR}/upscale_models",
    f"wget https://github.com/Phhofm/models/releases/download/2xPublic_realplksr_dysample_layernorm_real_nn/2xPublic_realplksr_dysample_layernorm_real_nn.pth -P {MODELS_DIR}/upscale_models",
 #   f"wget https://github.com/Phhofm/models/releases/download/2xParagonSR_Nano_gan/2xParagonSR_Nano_gan_op18_fp16.onnx -P {MODELS_DIR}/upscale_models",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/yoloface_8n.onnx -P {MODELS_DIR}/faceless/face_detector",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/2dfan4.onnx -P {MODELS_DIR}/faceless/face_landmarker",
    f"wget https://huggingface.co/crj/dl-ws/resolve/eb6e7e673da4c13994012ab57d71e94ba16f6a5a/face_landmarker_68_5.onnx -P {MODELS_DIR}/faceless/face_landmarker",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/arcface_w600k_r50.onnx -P {MODELS_DIR}/faceless/face_recognizer",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/gfpgan_1.4.onnx -P {MODELS_DIR}/faceless/face_restoration",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/gpen_bfr_512.onnx -P {MODELS_DIR}/faceless/face_restoration",
    f"wget https://github.com/Glat0s/GFPGAN-1024-onnx/releases/download/v0.0.1/gfpgan-1024.onnx -P {MODELS_DIR}/faceless/face_restoration",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/restoreformer_plus_plus.onnx -P {MODELS_DIR}/faceless/face_restoration",
    f"wget https://huggingface.co/crj/dl-ws/resolve/main/gender_age.onnx -P {MODELS_DIR}/faceless",
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
            print("Detected detached HEAD, checking out master branch...")
            subprocess.run("git checkout -B master origin/master", shell=True, check=True, capture_output=True, text=True)
            print("Successfully checked out master branch")
        # Configure pull strategy to fast-forward only
        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        # Perform git pull
        result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        print("Git pull output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI backend: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during backend update: {e}")

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

    # Configure ComfyUI-Manager: Disable auto-fetch, set weak security, and disable file logging
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    print("Configuring ComfyUI-Manager: Disabling auto-fetch, setting security_level to weak, and disabling file logging...")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)
    print(f"Updated {manager_config_path} with security_level=weak, log_to_file=false")

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
    
    # Install pip dependencies for new ComfyUI Manager
    print("Installing pip dependencies for new ComfyUI Manager...")
    manager_req_path = os.path.join(DATA_BASE, "manager_requirements.txt")
    if os.path.exists(manager_req_path):
        try:
            result = subprocess.run(
                f"pip install -r {manager_req_path}",
                shell=True, check=True, capture_output=True, text=True
            )
            print("New Manager dependencies installed:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error installing new Manager dependencies: {e.stderr}")
    else:
        print(f"Warning: {manager_req_path} not found, skipping new Manager dependencies installation")

    # Ensure all required directories exist
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # Download models at runtime (only if missing)
    print("Checking and downloading missing models...")
    for repo, fn, sub, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            print(f"Downloading {fn} to {target}...")
            try:
                hf_download(repo, fn, sub, subf)
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
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest", "--enable-manager"]
    print(f"Executing: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
