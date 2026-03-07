from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYBIND=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_PLUGINS=OFF",
        ]
        build_args = []

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", "tensorium"] + build_args,
                              cwd=self.build_temp)


setup(
    name="tensorium",
    version="0.1.0",
    package_dir={"": "Pysrc"},
    packages=find_packages(where="Pysrc"),
    ext_modules=[CMakeExtension("tensorium.tensorium")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
