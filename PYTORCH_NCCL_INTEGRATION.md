# PyTorch NCCL Integration

This document describes the PyTorch NCCL integration feature in vLLM that allows using NCCL statically from PyTorch's dependency instead of dynamically loading it.

## Overview

vLLM now supports two methods for using NCCL:

1. **Dynamic Loading** (original approach): Dynamically loads NCCL shared libraries at runtime
2. **PyTorch Integration** (new approach): Uses NCCL library bundled with PyTorch

## Benefits of PyTorch NCCL Integration

- **Eliminate dependency on system NCCL libraries**: No need to manage NCCL library paths
- **Version consistency**: Uses the exact NCCL version that PyTorch was compiled with
- **Simplified deployment**: Reduces runtime library discovery issues
- **Better integration**: Tighter integration with PyTorch's memory management

## Configuration

### Environment Variable

Set the `VLLM_USE_PYTORCH_NCCL` environment variable:

```bash
# Use PyTorch bundled NCCL
export VLLM_USE_PYTORCH_NCCL=true

# Use dynamic loading (original behavior)
export VLLM_USE_PYTORCH_NCCL=false

# Auto-detect (default) - prefers PyTorch NCCL if available
unset VLLM_USE_PYTORCH_NCCL
```

### Programmatic Configuration

When creating a `PyNcclCommunicator` directly:

```python
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

# Use PyTorch NCCL
comm = PyNcclCommunicator(group, device, use_pytorch_nccl=True)

# Use dynamic loading
comm = PyNcclCommunicator(group, device, use_pytorch_nccl=False)

# Auto-detect (default)
comm = PyNcclCommunicator(group, device)  # or use_pytorch_nccl=None
```

## Auto-Detection Behavior

When no explicit configuration is provided:

1. Check if PyTorch NCCL is available
2. If available, use PyTorch NCCL
3. If not available, fall back to dynamic loading
4. If both fail, disable NCCL functionality

## Fallback Mechanism

The implementation includes a robust fallback mechanism:

1. If PyTorch NCCL is requested but fails to initialize, automatically fall back to dynamic loading
2. If dynamic loading also fails, disable NCCL functionality gracefully
3. Log appropriate messages for debugging

## CUDA Graph Compatibility

Both approaches maintain CUDA graph compatibility, which was the original motivation for vLLM's custom NCCL wrapper.

## Testing

A test script is provided to verify the implementation:

```bash
cd /path/to/vllm
python test_pytorch_nccl.py
```

This tests:
- PyTorch NCCL availability detection
- Library initialization with both methods
- Auto-detection functionality
- Environment variable configuration
- Fallback mechanisms

## Migration Guide

### For Users

No changes are required. The new feature is enabled by default with auto-detection. If you encounter issues, you can:

1. Explicitly disable PyTorch NCCL: `export VLLM_USE_PYTORCH_NCCL=false`
2. Use the existing `VLLM_NCCL_SO_PATH` environment variable for custom library paths

### For Developers

The `PyNcclCommunicator` API remains backward compatible. New parameters:

- `use_pytorch_nccl`: Optional boolean to control NCCL loading method
- `_use_pytorch_nccl`: Internal attribute indicating which method is being used

## Implementation Details

### Key Files

- `vllm/distributed/device_communicators/pytorch_nccl_wrapper.py`: New PyTorch NCCL wrapper
- `vllm/distributed/device_communicators/pynccl.py`: Updated to support both methods
- `vllm/envs.py`: Added `VLLM_USE_PYTORCH_NCCL` environment variable
- `vllm/utils/__init__.py`: Updated documentation for `find_nccl_library`

### Architecture

The implementation maintains the same interface as the original dynamic loading approach:

1. **PyTorchNCCLLibrary**: Provides the same API as `NCCLLibrary` but uses PyTorch's NCCL
2. **Detection Function**: `is_pytorch_nccl_available()` checks if PyTorch NCCL can be loaded
3. **Unified Interface**: `PyNcclCommunicator` transparently chooses between implementations

## Troubleshooting

### PyTorch NCCL Not Available

If PyTorch NCCL is not available:
- Check if PyTorch was compiled with NCCL support
- Verify CUDA/ROCm support in your PyTorch installation
- Fall back to dynamic loading with `VLLM_USE_PYTORCH_NCCL=false`

### Library Loading Errors

If you encounter library loading errors:
1. Check PyTorch installation: `python -c "import torch; print(torch.version.cuda)"`
2. Verify NCCL availability: `python -c "from vllm.distributed.device_communicators.pytorch_nccl_wrapper import is_pytorch_nccl_available; print(is_pytorch_nccl_available())"`
3. Use dynamic loading as fallback
4. Check system NCCL installation and paths

### Environment Variable Not Working

If the environment variable doesn't seem to work:
- Ensure it's set before importing vLLM modules
- Check the value: `python -c "import vllm.envs; print(vllm.envs.VLLM_USE_PYTORCH_NCCL)"`
- Restart your Python process after setting the variable