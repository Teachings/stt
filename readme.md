# STT (Speech-to-Text) System with Faster-Whisper

A GPU-accelerated speech recognition system with server/client architecture.  
Press hotkeys to record → transcribe → output to clipboard/auto-type.

![Platform Support](https://img.shields.io/badge/Linux-✓-success) ![Platform Support](https://img.shields.io/badge/macOS-✓-success)

---

## 🚀 Quick Start (Both Platforms)

### 1. Conda Environment (Python 3.11)
```bash
conda create -n stt python=3.11
conda activate stt
```

### 2. GPU Setup (Skip for CPU-only)
```bash
conda install -c "nvidia/label/cuda-12.8.0" cudnn  # Must run BEFORE pip installs!
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

# Linux Setup

## Server Launch

```bash
python stt_server.py --start
```

Or if you want to run it on second GPU
```bash
CUDA_VISIBLE_DEVICES="1" python stt_server.py --start
```

## Client Setup
```bash
python stt_client.py --configure  # Set hotkey (default: Ctrl+Alt+Space)
python stt_client.py --start
```

### Linux-Specific Notes:
- Install `xdotool` for auto-typing:
  ```bash
  sudo apt install xdotool
  ```
- Audio troubleshooting:
  ```bash
  sudo apt install libportaudio2  # If sounddevice fails
  ```

---

# MacOS Setup

## Server Launch (Same as Linux)

## Client Setup
```bash
python stt_client_mac.py --configure  # Set hotkey (default: Cmd+Space)
python stt_client_mac.py --start
```

### Mac-Specific Requirements:
```bash
brew install portaudio  # For audio input
```

### Essential Permissions:
1. Enable in `System Settings > Privacy & Security > Accessibility`
2. Add your Python interpreter (from `which python`)

---

## ⚙️ Configuration

Edit the YAML files to customize:
```bash
~/.config/stt_server/config.yaml   # Server settings
~/.config/stt_client/config.yaml   # Linux client
~/.config/stt_client/config.yaml   # Mac client (same path)
```

### Key Options:
```yaml
# Server
use_gpu: true
compute_type: "float16"  # int8|float16|float32
model: "large-v3-turbo"

# Client
hotkey: "<ctrl>+<alt>+<space>"  # Linux
hotkey: "<cmd>+<space>"         # Mac
output_mode: "both"  # clipboard|type|both
```

---

## 📋 Requirements

### Hardware
- NVIDIA GPU recommended (CUDA 12.8+)
- Microphone

### `requirements.txt`
```
fastapi>=0.95.2
uvicorn>=0.21.1
faster-whisper>=0.7.1
torch>=2.0.0
sounddevice>=0.4.6
numpy>=1.24.3
pyyaml>=6.0
pynput>=1.7.6
pyperclip>=1.8.2
requests>=2.28.2
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA errors | Verify `nvcc --version` matches Conda's CUDA |
| "Invalid handle" | Reinstall `cudnn` via Conda before other packages |
| Auto-type fails | On Mac: check Accessibility permissions<br>On Linux: install `xdotool` |
| Low GPU usage | Try `compute_type="int8"` or smaller model |

---

## 🏗️ Architecture

```
stt_server.py        # FastAPI service (GPU processing)
stt_client.py        # Linux hotkey client
stt_client_mac.py    # Mac-optimized client
```

---


### Key Improvements:
1. **Platform Separation**: Clear Linux/Mac sections with OS-specific instructions
2. **Conda-First Approach**: GPU setup instructions precede pip installs
3. **Visual Enhancements**: Badges, tables, and clean formatting
4. **Problem-Solution Pairs**: Structured troubleshooting table
5. **Configuration Highlight**: Important YAML options shown inline
6. **Architecture Overview**: Simple filesystem structure explanation

Choose this version if you want:
- Faster onboarding with platform-specific instructions
- Emphasis on Conda for GPU support
- Visual clarity through badges and tables
- Quick troubleshooting reference