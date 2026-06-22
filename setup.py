from setuptools import setup, Extension
import pybind11

# Set up files
source_files = [
    "python-packages/bindings.cpp", 
    "src/engine/XGBoost.cpp" 
]

ext_modules = [
    Extension(
        "my_xgboost", # Name used when using import
        source_files,
        include_dirs=[
            pybind11.get_include(), 
            "include",
            "src"
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-Wall"] # These flags tell the compiler to heavily optimize the C++ for speed
    ),
]

# Executing the build
setup(
    name="my_xgboost",
    version="1.0.0",
    description="Custom high-performance C++ XGBoost implementation",
    ext_modules=ext_modules,
    zip_safe=False,
)