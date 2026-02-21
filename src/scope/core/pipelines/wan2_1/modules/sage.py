# Modified from https://github.com/krea-ai/realtime-video
import os

import torch

SAGEATTN_AVAILABLE = False
try:
    if os.getenv("DISABLE_SAGEATTENTION", "0") != "0":
        raise Exception("DISABLE_SAGEATTENTION is set")

    from sageattn3 import sageattn3_blackwell

    @torch.library.custom_op(
        "mylib::sageattn", mutates_args={"q", "k", "v"}, device_types="cuda"
    )
    def sageattn_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        return sageattn3_blackwell(
            q,
            k,
            v,
            is_causal=is_causal,
        )

    @sageattn_func.register_fake
    def _sageattn_fake(q, k, v, is_causal=False, sm_scale=None):
        return torch.empty(*q.shape, device=q.device, dtype=q.dtype)

    # Runtime kernel probe: verify the AOT-compiled CUDA kernels support this GPU.
    # The import can succeed even when kernels are incompatible (e.g. precompiled
    # wheel missing sm_89). The error is async, so torch.cuda.synchronize() is
    # needed to surface it.
    if torch.cuda.is_available():
        _q = torch.randn(1, 16, 64, 128, dtype=torch.bfloat16, device="cuda")
        _k = torch.randn(1, 16, 64, 128, dtype=torch.bfloat16, device="cuda")
        _v = torch.randn(1, 16, 64, 128, dtype=torch.bfloat16, device="cuda")
        try:
            sageattn_func(_q, _k, _v)
            torch.cuda.synchronize()
        except Exception as probe_err:
            raise RuntimeError(
                "SageAttention kernel probe failed — kernels are not "
                f"compatible with this GPU: {probe_err}"
            ) from probe_err
        finally:
            del _q, _k, _v

        # CUDA health check: verify the CUDA context is still functional after
        # loading SageAttention's native extensions.  Some builds register
        # kernels whose deferred errors only surface on the next CUDA call.
        _a = _b = None
        try:
            _a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
            _b = _a @ _a.T
            torch.cuda.synchronize()
        except Exception as health_err:
            raise RuntimeError(
                "CUDA health check failed after loading SageAttention — "
                f"native extensions are incompatible with this GPU: {health_err}"
            ) from health_err
        finally:
            del _a, _b

    print("SageAttention 3 loaded successfully")

    SAGEATTN_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load sageattention: {str(e)}")
    if isinstance(e, ModuleNotFoundError):
        print("sageattn3 package is not installed")
    elif isinstance(e, ImportError) and "DLL" in str(e):
        print("sageattn3 DLL loading error")
    elif "kernel probe failed" in str(e) or "health check failed" in str(e):
        print("sageattn3 kernels are not compatible with this GPU.")
    sageattn_func = None
