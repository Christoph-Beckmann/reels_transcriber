#!/bin/bash
# Unified Installation Script for Instagram Reels Transcriber
# One script to install everything including Python if needed

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║     Instagram Reels Transcriber - Complete Installation         ║"
echo "║                                                                  ║"
echo "║     One command to install EVERYTHING you need!                 ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)
echo "🔍 Detected OS: $OS_TYPE"
echo

# Step 1: Check/Install Python 3.12
echo "📦 Step 1: Python 3.12"
echo "────────────────────"
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️  Python 3.12 not found. Installing..."

    case $OS_TYPE in
        macOS)
            # Install Homebrew if needed
            if ! command -v brew &> /dev/null; then
                echo "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            echo "Installing Python 3.12 via Homebrew..."
            brew install python@3.12
            brew install python-tk@3.12  # For GUI support
            ;;
        linux)
            if command -v apt-get &> /dev/null; then
                echo "Installing Python 3.12 via apt..."
                sudo apt-get update
                sudo apt-get install -y python3.12 python3.12-venv python3.12-tk
            elif command -v yum &> /dev/null; then
                echo "Installing Python 3.12 via yum..."
                sudo yum install -y python312 python312-tkinter
            else
                echo "❌ Unsupported Linux distribution"
                echo "Please install Python 3.12 manually"
                exit 1
            fi
            ;;
        windows)
            echo "Windows detected. Please install Python 3.12 from:"
            echo "https://www.python.org/downloads/"
            exit 1
            ;;
        *)
            echo "❌ Unknown OS. Please install Python 3.12 manually"
            exit 1
            ;;
    esac

    # Verify installation
    if ! command -v python3.12 &> /dev/null; then
        echo "❌ Python 3.12 installation failed"
        exit 1
    fi
fi
echo "✅ Python 3.12: $(python3.12 --version)"
echo

# Step 2: Install uv package manager
echo "📦 Step 2: UV Package Manager"
echo "─────────────────────────────"
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source shell config to get uv in PATH
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc"
    fi

    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo "⚠️  uv installed but not in PATH. Please run:"
        echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\""
        echo "  Then re-run this script"
        exit 1
    fi
fi
echo "✅ uv: $(uv --version)"
echo

# Step 3: Create virtual environment
echo "📦 Step 3: Virtual Environment"
echo "──────────────────────────────"
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi
echo "Creating new virtual environment with Python 3.12..."
uv venv --python python3.12
echo "✅ Virtual environment created"
echo

# Step 4: Install dependencies
echo "📦 Step 4: Python Dependencies"
echo "──────────────────────────────"
echo "Installing all required packages..."

# Create comprehensive requirements file if it doesn't exist
cat > requirements_complete.txt << 'EOF'
# Core functionality
FreeSimpleGUI==5.2.0
instaloader==4.14.2
yt-dlp>=2024.1.0
moviepy==1.0.3
faster-whisper>=1.2.0

# AI Models
openai-whisper>=20230918

# Audio processing
numpy>=1.26.0
scipy>=1.14.0

# System utilities
psutil==5.9.5
requests>=2.32.0
urllib3>=2.0.0

# Logging and configuration
python-dotenv==1.0.0

# Testing
pytest-benchmark>=4.0.0

# UI
rich>=13.0.0
EOF

source .venv/bin/activate
uv pip install -r requirements_complete.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully"
else
    echo "⚠️  Some dependencies failed to install"
fi
echo

# Step 5: Download Whisper model
echo "📦 Step 5: Whisper AI Model"
echo "───────────────────────────"
python3.12 << 'PYTHON_SCRIPT'
import urllib.request
import hashlib
from pathlib import Path
import sys
import time

# Define MODELS dictionary first
MODELS = {
    'tiny': {
        'url': 'https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt',
        'size_mb': 39,
        'sha256': '65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9'
    },
    'base': {
        'url': 'https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt',
        'size_mb': 74,
        'sha256': 'ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e'
    },
    'small': {
        'url': 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt',
        'size_mb': 244,
        'sha256': '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794'
    },
    'medium': {
        'url': 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt',
        'size_mb': 769,
        'sha256': '345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1'
    },
    'large': {
        'url': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6884409c/large-v2.pt',
        'size_mb': 1550,
        'sha256': '81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6884409c'
    }
}

# Model selection - Interactive menu
print("\n" + "="*60)
print("🧠 Choose Whisper AI Model Size")
print("="*60)
print("Different model sizes offer different trade-offs between speed, accuracy, and disk space:")
print()
print("1. 🚀 Tiny    (39MB)   - Fastest, good for testing")
print("2. ⚡ Base    (74MB)   - Good balance (recommended)")
print("3. 📈 Small   (244MB)  - Better accuracy, slower")
print("4. 🎯 Medium  (769MB)  - High accuracy, much slower")
print("5. 🏆 Large   (1550MB) - Best accuracy, very slow")
print()
print("💡 Recommendation: Choose 'Base' for most users, 'Large' for best quality")
print("="*60)

# Check if we should force non-interactive mode (for CI/CD environments)
import os
import sys

# Force non-interactive if specific environment variables are set OR if there's no TTY
force_non_interactive = (
    os.environ.get('CI') == 'true' or  # GitHub Actions, GitLab CI, etc.
    os.environ.get('GITHUB_ACTIONS') == 'true' or
    os.environ.get('GITLAB_CI') == 'true' or
    os.environ.get('JENKINS_URL') is not None or
    os.environ.get('BUILDKITE') == 'true' or
    os.environ.get('CIRCLECI') == 'true' or
    os.environ.get('WHISPER_MODEL_AUTO') is not None  # Allow manual override
)

if force_non_interactive:
    # Auto-select model for CI/CD
    MODEL_SIZE = os.environ.get('WHISPER_MODEL_AUTO', 'base')
    print(f"🤖 CI/CD environment detected - using {MODEL_SIZE} model")
    print("💡 Set WHISPER_MODEL_AUTO=tiny|base|small|medium|large to choose different model")
else:
    # Interactive mode: show menu and get user choice
    print("📝 Please make your selection...")
    try:
        choice = input("Enter your choice (1-5) [default: 2 for Base]: ").strip()

        # Default to base if nothing entered
        if not choice:
            choice = "2"

        choice_num = int(choice)

        if choice_num == 1:
            MODEL_SIZE = "tiny"
            print(f"✅ Selected: Tiny model (39MB) - Fast processing")
        elif choice_num == 2:
            MODEL_SIZE = "base"
            print(f"✅ Selected: Base model (74MB) - Recommended balance")
        elif choice_num == 3:
            MODEL_SIZE = "small"
            print(f"✅ Selected: Small model (244MB) - Better accuracy")
        elif choice_num == 4:
            MODEL_SIZE = "medium"
            print(f"✅ Selected: Medium model (769MB) - High accuracy")
        elif choice_num == 5:
            MODEL_SIZE = "large"
            print(f"✅ Selected: Large model (1550MB) - Best accuracy")
        else:
            print("❌ Invalid choice, using default Base model")
            MODEL_SIZE = "base"

    except (ValueError, EOFError, KeyboardInterrupt) as e:
        print(f"\n🤖 Input error ({type(e).__name__}) - using default Base model (74MB)")
        print("💡 For automated installations, set WHISPER_MODEL_AUTO=model_name")
        MODEL_SIZE = "base"

print(f"\n🎯 Will download {MODEL_SIZE} model ({MODELS[MODEL_SIZE]['size_mb']}MB)")

# Save model selection to configuration file
import json

config_data = {
    "whisper_model": MODEL_SIZE
}

config_path = Path("config.json")
try:
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"💾 Saved model selection to {config_path}")
except Exception as e:
    print(f"⚠️  Warning: Could not save config ({e}), but continuing installation...")

print()

model_info = MODELS[MODEL_SIZE]

# Determine cache directory
import os
if os.name == 'nt':  # Windows
    cache_dir = Path.home() / 'AppData' / 'Local' / 'whisper'
else:  # macOS/Linux
    cache_dir = Path.home() / '.cache' / 'whisper'

cache_dir.mkdir(parents=True, exist_ok=True)
model_path = cache_dir / f'{MODEL_SIZE}.pt'

# Check if already downloaded
if model_path.exists():
    print(f"✅ Model '{MODEL_SIZE}' already exists at {model_path}")
    # Verify checksum
    sha256_hash = hashlib.sha256()
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)

    if sha256_hash.hexdigest() == model_info['sha256']:
        print("✅ Model verified successfully")
        sys.exit(0)
    else:
        print("⚠️  Model corrupted, re-downloading...")

# Download the model
print(f"Downloading {MODEL_SIZE} model ({model_info['size_mb']}MB)...")
temp_path = model_path.with_suffix('.download')

def download_with_progress(url, dest):
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get('Content-Length', 0))
    downloaded = 0
    block_size = 8192

    with open(dest, 'wb') as f:
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            f.write(buffer)

            # Show progress
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f}MB)", end='', flush=True)
    print()  # New line after progress

try:
    download_with_progress(model_info['url'], temp_path)

    # Verify download
    sha256_hash = hashlib.sha256()
    with open(temp_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)

    if sha256_hash.hexdigest() == model_info['sha256']:
        temp_path.rename(model_path)
        print(f"✅ Model downloaded and verified successfully")
    else:
        temp_path.unlink(missing_ok=True)
        print("❌ Model verification failed")
        sys.exit(1)

except Exception as e:
    print(f"❌ Download failed: {e}")
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)
    sys.exit(1)
PYTHON_SCRIPT
echo

# Step 6: Test installation
echo "📦 Step 6: Testing Installation"
echo "───────────────────────────────"
source .venv/bin/activate
python -c "
import sys
errors = []

try:
    import whisper
    print('✅ Whisper AI model')
except ImportError as e:
    errors.append(f'❌ Whisper: {e}')

try:
    import FreeSimpleGUI
    print('✅ GUI framework')
except ImportError as e:
    errors.append(f'⚠️  GUI framework (tkinter may be missing): {e}')

try:
    import yt_dlp
    print('✅ YouTube downloader')
except ImportError as e:
    errors.append(f'❌ YouTube downloader: {e}')

try:
    import moviepy
    print('✅ Video processing')
except ImportError as e:
    errors.append(f'❌ Video processing: {e}')

try:
    import faster_whisper
    print('✅ Faster Whisper')
except ImportError as e:
    errors.append(f'❌ Faster Whisper: {e}')

if errors:
    print()
    for error in errors:
        print(error)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║     🎉 Installation Complete! 🎉                                ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo
    echo "📚 Next Steps:"
    echo "─────────────"
    echo
    echo "1. Activate the virtual environment:"
    echo "   source .venv/bin/activate"
    echo
    echo "2. Run the application:"
    echo "   • GUI Mode: python main.py"
    echo "   • CLI Mode: python cli.py --help"
    echo
    echo "3. Quick test with a URL:"
    echo "   python cli.py --url 'YOUR_INSTAGRAM_REEL_URL'"
    echo
    echo "Enjoy transcribing your Instagram Reels! 🎬"
else
    echo
    echo "⚠️  Installation completed with some warnings"
    echo "The application may still work, but some features might be limited"
fi
