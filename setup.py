import os
from setuptools import setup, find_packages

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

common_setup_kwargs = {
    "version": "0.2.0",
    "name": "auto_auto_gptq",
    "author": "PanQiWei",
    "description": "An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.",
    "url": "https://github.com/PanQiWei/AutoGPTQ",
    "keywords": ["gptq", "quantization", "large-language-models", "pytorch", "transformers"],
    "platforms": ["windows", "linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.7",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Framework :: pytorch",
        "Framework :: transformers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
    "python_requires": ">=3.8"
}

requirements = [
    "accelerate>=0.19.0",
    "datasets",
    "numpy",
    "rouge",
    "torch>=1.13.0",
    "safetensors",
    "transformers>=4.26.1"
]

extras_require = {
    "llama": ["transformers>=4.28.0"],
    "triton": ["triton>=2.0.0"]
}

include_dirs = ["autogptq_cuda"]

if TORCH_AVAILABLE:
    BUILD_CUDA_EXT = int(os.environ.get('BUILD_CUDA_EXT', '1')) == 1
    
    additional_setup_kwargs = dict()
    if BUILD_CUDA_EXT and torch.cuda.is_available():
        from torch.utils import cpp_extension
        from distutils.sysconfig import get_python_lib
        conda_cuda_include_dir=os.path.join(get_python_lib(),"nvidia/cuda_runtime/include")
        if os.path.isdir(conda_cuda_include_dir):
            include_dirs.append(conda_cuda_include_dir)
            print(f"appending conda cuda include dir {conda_cuda_include_dir}")
        extensions = [
            cpp_extension.CUDAExtension(
                "autogptq_cuda",
                [
                    "autogptq_cuda/autogptq_cuda.cpp",
                    "autogptq_cuda/autogptq_cuda_kernel.cu"
                ]
            )
        ]

        additional_setup_kwargs = {
            "ext_modules": extensions,
            "cmdclass": {'build_ext': cpp_extension.BuildExtension}
        }
    common_setup_kwargs.update(additional_setup_kwargs)
    setup(
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=include_dirs,
        **common_setup_kwargs
    )
else:
    setup(
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras_require,
        include_dirs=include_dirs,
        **common_setup_kwargs
    )
