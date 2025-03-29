import pybind11
from setuptools import Extension, setup

setup(
    name="liblocalization_cpp",
    version="0.0.0",
    packages=["_liblocalization_cpp"],
    ext_modules=[
        Extension(
            "_liblocalization_cpp",
            [
                "distance_transform.cpp",
                "bindings.cpp",
            ],
            include_dirs=[pybind11.get_include()],
            language="c++",
        ),
    ],
    zip_safe=False,
)
