# --- Step 0: Set your virtual environment path ---
$venvPath = "C:\users\Julian\violence-detection-fbf\venv310"
$pythonExe = "$venvPath\Scripts\python.exe"

if (-Not (Test-Path $pythonExe)) {
    Write-Host "Python executable not found at $pythonExe"
    exit 1
} else {
    Write-Host "Using existing virtual environment at $venvPath"
}

# --- Step 1: Upgrade pip ---
Write-Host ">>> Step 1: Upgrading pip..."
& $pythonExe -m pip install --upgrade pip

# --- Step 2: Remove old TensorFlow, Keras, and supporting packages ---
Write-Host ">>> Step 2: Removing old TensorFlow, Keras, and supporting packages..."
$oldPackages = @(
    "tensorflow",
    "tensorflow-gpu",
    "tf-nightly",
    "keras",
    "keras-nightly",
    "protobuf",
    "tensorboard",
    "tb-nightly",
    "tensorboard-data-server"
)
foreach ($pkg in $oldPackages) {
    & $pythonExe -m pip uninstall -y $pkg
}

# --- Step 3: Install TensorFlow GPU (safe version for CUDA 11.8) ---
Write-Host ">>> Step 3: Installing TensorFlow GPU..."
& $pythonExe -m pip install tensorflow==2.13.1

# --- Step 4: Install supporting packages ---
Write-Host ">>> Step 4: Installing supporting packages..."
$supportPackages = @(
    "keras==2.13.1",
    "h5py",
    "absl-py",
    "numpy>=1.22,<1.25",
    "rich",
    "ml-dtypes",
    "optree",
    "packaging"
)
foreach ($pkg in $supportPackages) {
    & $pythonExe -m pip install $pkg
}

# --- Step 5: Reminder about NVIDIA driver and CUDA 11.8 ---
Write-Host ">>> Step 5: Ensure your NVIDIA driver is up-to-date and CUDA 11.8 installed."
Write-Host "Check with: nvidia-smi"

# --- Step 6: Test TensorFlow GPU detection ---
Write-Host ">>> Step 6: Testing TensorFlow GPU detection..."
$gpuTest = @"
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU(s) detected:", gpus)
else:
    print("❌ No GPU detected")

try:
    from tensorflow.python.platform import build_info as tf_build_info
    print("Build info - CUDA:", tf_build_info.build_info.get('cuda_version','Unknown'), ", cuDNN:", tf_build_info.build_info.get('cudnn_version','Unknown'))
except:
    print("Build info - CUDA and cuDNN: Unknown")
"@

$tempPy = "$env:TEMP\tf_gpu_test.py"
$gpuTest | Out-File -FilePath $tempPy -Encoding UTF8
& $pythonExe $tempPy
Remove-Item $tempPy

Write-Host ">>> Setup complete!"
