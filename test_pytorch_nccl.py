#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test script to verify PyTorch NCCL integration functionality.

This script tests:
1. PyTorch NCCL library availability detection
2. Library initialization with both PyTorch and dynamic loading
3. Basic NCCL functionality (if CUDA is available)
"""

import os
import sys
import traceback

def test_pytorch_nccl_availability():
    """Test if PyTorch NCCL is available."""
    print("=" * 60)
    print("Testing PyTorch NCCL availability...")
    try:
        from vllm.distributed.device_communicators.pytorch_nccl_wrapper import is_pytorch_nccl_available
        available = is_pytorch_nccl_available()
        print(f"PyTorch NCCL available: {available}")
        return available
    except Exception as e:
        print(f"Error checking PyTorch NCCL availability: {e}")
        traceback.print_exc()
        return False

def test_pytorch_nccl_library():
    """Test PyTorch NCCL library initialization."""
    print("=" * 60)
    print("Testing PyTorch NCCL library initialization...")
    try:
        from vllm.distributed.device_communicators.pytorch_nccl_wrapper import PyTorchNCCLLibrary
        lib = PyTorchNCCLLibrary()
        version = lib.ncclGetVersion()
        print(f"PyTorch NCCL version: {version}")
        return True
    except Exception as e:
        print(f"Error initializing PyTorch NCCL library: {e}")
        traceback.print_exc()
        return False

def test_dynamic_nccl_library():
    """Test dynamic NCCL library initialization."""
    print("=" * 60)
    print("Testing dynamic NCCL library initialization...")
    try:
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
        lib = NCCLLibrary()
        version = lib.ncclGetVersion()
        print(f"Dynamic NCCL version: {version}")
        return True
    except Exception as e:
        print(f"Error initializing dynamic NCCL library: {e}")
        traceback.print_exc()
        return False

def test_pynccl_communicator_pytorch():
    """Test PyNcclCommunicator with PyTorch NCCL."""
    print("=" * 60)
    print("Testing PyNcclCommunicator with PyTorch NCCL...")
    try:
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Create a mock stateless process group for single node testing
        group = StatelessProcessGroup(rank=0, world_size=1)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Test with PyTorch NCCL explicitly enabled
        comm = PyNcclCommunicator(group, device, use_pytorch_nccl=True)
        print(f"PyNcclCommunicator created successfully with PyTorch NCCL")
        print(f"NCCL version: {comm.nccl.ncclGetVersion() if hasattr(comm, 'nccl') else 'N/A'}")
        print(f"Available: {comm.available}, Disabled: {comm.disabled}")
        return True
    except Exception as e:
        print(f"Error testing PyNcclCommunicator with PyTorch NCCL: {e}")
        traceback.print_exc()
        return False

def test_pynccl_communicator_dynamic():
    """Test PyNcclCommunicator with dynamic NCCL."""
    print("=" * 60)
    print("Testing PyNcclCommunicator with dynamic NCCL...")
    try:
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Create a mock stateless process group for single node testing
        group = StatelessProcessGroup(rank=0, world_size=1)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Test with dynamic NCCL explicitly enabled
        comm = PyNcclCommunicator(group, device, use_pytorch_nccl=False)
        print(f"PyNcclCommunicator created successfully with dynamic NCCL")
        print(f"NCCL version: {comm.nccl.ncclGetVersion() if hasattr(comm, 'nccl') else 'N/A'}")
        print(f"Available: {comm.available}, Disabled: {comm.disabled}")
        return True
    except Exception as e:
        print(f"Error testing PyNcclCommunicator with dynamic NCCL: {e}")
        traceback.print_exc()
        return False

def test_pynccl_communicator_auto():
    """Test PyNcclCommunicator with auto-detection."""
    print("=" * 60)
    print("Testing PyNcclCommunicator with auto-detection...")
    try:
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Create a mock stateless process group for single node testing
        group = StatelessProcessGroup(rank=0, world_size=1)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Test with auto-detection (default behavior)
        comm = PyNcclCommunicator(group, device)
        print(f"PyNcclCommunicator created successfully with auto-detection")
        print(f"Using PyTorch NCCL: {comm._use_pytorch_nccl}")
        print(f"NCCL version: {comm.nccl.ncclGetVersion() if hasattr(comm, 'nccl') else 'N/A'}")
        print(f"Available: {comm.available}, Disabled: {comm.disabled}")
        return True
    except Exception as e:
        print(f"Error testing PyNcclCommunicator with auto-detection: {e}")
        traceback.print_exc()
        return False

def test_environment_variable():
    """Test environment variable configuration."""
    print("=" * 60)
    print("Testing environment variable configuration...")
    try:
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        # Set environment variable to force PyTorch NCCL
        os.environ["VLLM_USE_PYTORCH_NCCL"] = "true"

        # Import envs after setting the environment variable to force re-evaluation
        import importlib
        import vllm.envs
        importlib.reload(vllm.envs)

        group = StatelessProcessGroup(rank=0, world_size=1)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Test with environment variable set
        comm = PyNcclCommunicator(group, device)
        print(f"Environment variable test: Using PyTorch NCCL: {comm._use_pytorch_nccl}")

        # Clean up
        if "VLLM_USE_PYTORCH_NCCL" in os.environ:
            del os.environ["VLLM_USE_PYTORCH_NCCL"]

        return True
    except Exception as e:
        print(f"Error testing environment variable configuration: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("vLLM PyTorch NCCL Integration Test")
    print("=" * 60)

    tests = [
        ("PyTorch NCCL Availability", test_pytorch_nccl_availability),
        ("PyTorch NCCL Library", test_pytorch_nccl_library),
        ("Dynamic NCCL Library", test_dynamic_nccl_library),
        ("PyNcclCommunicator with PyTorch NCCL", test_pynccl_communicator_pytorch),
        ("PyNcclCommunicator with Dynamic NCCL", test_pynccl_communicator_dynamic),
        ("PyNcclCommunicator with Auto-detection", test_pynccl_communicator_auto),
        ("Environment Variable Configuration", test_environment_variable),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False

    print("=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    # Return 0 if all critical tests pass
    critical_tests = [
        "PyTorch NCCL Availability",
        "PyNcclCommunicator with Auto-detection"
    ]

    all_critical_pass = all(results.get(test, False) for test in critical_tests)
    return 0 if all_critical_pass else 1

if __name__ == "__main__":
    sys.exit(main())