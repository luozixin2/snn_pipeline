import os, sys, subprocess
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

def find_cuda_home():
    # 1) 环境变量优先
    for k in ("CUDA_HOME", "CUDA_PATH"):
        if k in os.environ:
            return os.environ[k]
    # 2) 尝试 where nvcc (Windows) or which nvcc (Linux)
    try:
        if sys.platform == "win32":
            nvcc = subprocess.check_output(["where", "nvcc"], stderr=subprocess.DEVNULL)
        else:
            nvcc = subprocess.check_output(["which", "nvcc"], stderr=subprocess.DEVNULL)
        nvcc = nvcc.decode().strip().splitlines()[0]
        return os.path.dirname(os.path.dirname(nvcc))
    except:
        return None

cuda_home = find_cuda_home()
if cuda_home is None:
    print("Error: 找不到 CUDA Toolkit。请安装或设置 CUDA_HOME/CUDA_PATH。", file=sys.stderr)
    sys.exit(1)

setup(
    name="snn_pipeline",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="spike_cuda",
            sources=["snn_pipeline/kernels/spike_kernels.cu"],
            include_dirs=[os.path.join(cuda_home, "include")],
            library_dirs=[os.path.join(cuda_home, "lib", "x64")],  # Windows x64
            libraries=["cudart"],
            extra_compile_args={
                "cxx": [],
                "nvcc": ["-O3", "--use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=1.8.0"],
    zip_safe=False,
)