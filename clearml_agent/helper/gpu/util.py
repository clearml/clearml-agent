import os


_rocm_env_cache = None


def detect_rocm_env():
    # Host has an AMD ROCm GPU and no NVIDIA GPU.
    global _rocm_env_cache
    if _rocm_env_cache is not None:
        return _rocm_env_cache
    try:
        has_amd = os.path.exists('/dev/kfd')
        has_nvidia = os.path.exists('/dev/nvidia0') or os.path.exists('/proc/driver/nvidia')
        _rocm_env_cache = bool(has_amd and not has_nvidia)
    except Exception:
        _rocm_env_cache = False
    return _rocm_env_cache
