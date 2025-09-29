# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PyTorch-integrated NCCL wrapper that uses NCCL from PyTorch's dependency.

This wrapper provides the same interface as the dynamic NCCL wrapper but uses
PyTorch's bundled NCCL library instead of dynamically loading it. This approach:
1. Eliminates the need to find NCCL library paths at runtime
2. Uses the exact NCCL version that PyTorch was compiled with
3. Maintains CUDA graph compatibility
4. Provides better integration with PyTorch's memory management
"""

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger

logger = init_logger(__name__)

# Re-export types from the original wrapper for compatibility
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
ncclWindow_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


def _find_pytorch_nccl_library() -> str:
    """
    Find the NCCL library that PyTorch was compiled with.

    This function attempts to locate the NCCL library by:
    1. Checking if PyTorch has NCCL symbols loaded
    2. Looking in PyTorch's library directories
    3. Using standard library names that PyTorch would load
    """
    # First, try to get the library name based on PyTorch's CUDA/ROCm support
    if torch.version.cuda is not None:
        library_name = "libnccl.so.2"
    elif torch.version.hip is not None:
        library_name = "librccl.so.1"
    else:
        raise ValueError("PyTorch NCCL support requires CUDA or ROCm backend.")

    # Try to load the library name directly - this works if PyTorch
    # has already loaded NCCL into the process space
    return library_name


class PyTorchNCCLLibrary:
    """
    NCCL library wrapper that uses PyTorch's bundled NCCL dependency.

    This class provides the same interface as NCCLLibrary but leverages
    PyTorch's NCCL installation instead of dynamically loading it.
    """

    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t,
                 [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        Function("ncclCommInitRank", ncclResult_t, [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId,
            ctypes.c_int
        ]),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        Function("ncclAllReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, int root,
        #   ncclComm_t comm,  cudaStream_t stream);
        Function("ncclReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ctypes.c_int, ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        Function("ncclAllGather", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        Function("ncclReduceScatter", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ncclRedOp_t, ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function("ncclSend", ncclResult_t, [
            buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int,
            ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function("ncclRecv", ncclResult_t, [
            buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int,
            ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t ncclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, int root, ncclComm_t comm,
        #   cudaStream_t stream);
        Function("ncclBroadcast", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ctypes.c_int, ncclComm_t, cudaStream_t
        ]),
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
        # ncclResult_t ncclGroupStart();
        Function("ncclGroupStart", ncclResult_t, []),
        # ncclResult_t ncclGroupEnd();
        Function("ncclGroupEnd", ncclResult_t, []),
        # ncclResult_t ncclCommWindowRegister(
        #   ncclComm_t comm, void* buff, size_t size,
        #   ncclWindow_t* win, int winFlags);
        Function(
            "ncclCommWindowRegister",
            ncclResult_t,
            [
                ncclComm_t,
                buffer_type,
                ctypes.c_size_t,
                ctypes.POINTER(ncclWindow_t),
                ctypes.c_int,
            ],
        ),
        # ncclResult_t ncclCommWindowDeregister(
        #   ncclComm_t comm, ncclWindow_t win);
        Function("ncclCommWindowDeregister", ncclResult_t,
                 [ncclComm_t, ncclWindow_t]),
    ]

    # Class attribute to cache the library instance
    _library_cache: Optional['PyTorchNCCLLibrary'] = None
    _library_functions: Optional[dict[str, Any]] = None

    def __init__(self):
        """Initialize the PyTorch-based NCCL library wrapper."""

        # Use class-level caching to avoid reloading
        if PyTorchNCCLLibrary._library_cache is not None:
            self.lib = PyTorchNCCLLibrary._library_cache.lib
            self._funcs = PyTorchNCCLLibrary._library_functions
            return

        try:
            library_name = _find_pytorch_nccl_library()
            self.lib = ctypes.CDLL(library_name)
            logger.info(f"Successfully loaded PyTorch NCCL library: {library_name}")
        except Exception as e:
            logger.error(
                "Failed to load PyTorch NCCL library. "
                "This usually indicates PyTorch was not compiled with NCCL support "
                "or NCCL is not available in the current environment. "
                "Error: %s", e)
            raise e

        # Bind all NCCL functions
        self._funcs: dict[str, Any] = {}
        for func in self.exported_functions:
            try:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                self._funcs[func.name] = f
            except AttributeError:
                logger.warning(f"NCCL function {func.name} not found in library")
                # Some functions might not be available in older NCCL versions
                continue

        # Cache the successfully initialized library
        PyTorchNCCLLibrary._library_cache = self
        PyTorchNCCLLibrary._library_functions = self._funcs

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetRawVersion(self) -> int:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        return version.value

    def ncclGetVersion(self) -> str:
        version_str = str(self.ncclGetRawVersion())
        # Format version string (e.g., 21903 -> "2.19.3")
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](
            ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> ncclUniqueId:
        if len(data) != 128:
            raise ValueError(
                f"Expected 128 bytes for ncclUniqueId, got {len(data)} bytes")
        unique_id = ncclUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 128)
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId,
                         rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def ncclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def ncclReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                   count: int, datatype: int, op: int, root: int,
                   comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclReduce"](sendbuff, recvbuff, count,
                                                  datatype, op, root, comm,
                                                  stream))

    def ncclReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, op: int, comm: ncclComm_t,
                          stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclReduceScatter"](sendbuff, recvbuff,
                                                         count, datatype, op,
                                                         comm, stream))

    def ncclAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, stream))

    def ncclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def ncclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: ncclComm_t, stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def ncclBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, root: int, comm: ncclComm_t,
                      stream: cudaStream_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclBroadcast"](sendbuff, recvbuff, count,
                                                     datatype, root, comm,
                                                     stream))

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupEnd"]())

    def ncclCommWindowRegister(self, comm: ncclComm_t, buff: buffer_type,
                               size: int, win_flags: int) -> ncclWindow_t:
        window = ncclWindow_t()
        self.NCCL_CHECK(self._funcs["ncclCommWindowRegister"](
            comm, buff, size, ctypes.byref(window), win_flags))
        return window

    def ncclCommWindowDeregister(self, comm: ncclComm_t,
                                 window: ncclWindow_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommWindowDeregister"](comm, window))


def is_pytorch_nccl_available() -> bool:
    """Check if PyTorch NCCL library is available."""
    try:
        PyTorchNCCLLibrary()
        return True
    except Exception:
        return False


__all__ = [
    "PyTorchNCCLLibrary", "ncclDataTypeEnum", "ncclRedOpTypeEnum",
    "ncclUniqueId", "ncclComm_t", "cudaStream_t", "buffer_type",
    "is_pytorch_nccl_available"
]