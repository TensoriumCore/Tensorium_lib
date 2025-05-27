from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os, subprocess, sys


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
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = []

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--target", "tensorium"] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, ext_name):
        return f"{ext_name}.cpython-{sys.version_info.major}{sys.version_info.minor}-darwin.so"

setup(
    name="tensorium",
    version="0.1.0",
    package_dir={"": "pysrc"},
    packages=find_packages(where="pysrc"),
    ext_modules=[CMakeExtension("tensorium")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
