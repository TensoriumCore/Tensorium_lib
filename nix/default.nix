{
  pkgs ? import <nixpkgs> { },
  self,
  lib,
  llvmPkgs ? pkgs.llvmPackages_17,
  ...
}:

pkgs.stdenv.mkDerivation {
  name = "libTensorium";
  version = "0.1.0";
  src = lib.cleanSource self;

  nativeBuildInputs =
    with pkgs;
    [
      gnumake
    ]
    ++ (with llvmPkgs; [
      clang
      libclang
      openmp
      llvm
    ]);

  CXXFLAGS = "-fPIC";

  buildPhase = ''
    make lib
  '';

  installPhase = ''
       	mkdir -p $out/lib
      	find . -name '*.so' -exec cp -v {} $out/lib/ \;

      	mkdir -p $out/include
		cd includes/
    find . \( -name '*.h' -o -name '*.hpp' \) -exec cp --parents {} $out/include/ \;
  '';

  meta = with lib; {
    description = "An optimized Tensor/Matrix library for HPC applications an numerical relativity (AVX2/AVX512 and thread safe) with a Symbolic parser and Python wrapper";
    homepage = "https://at0m741.github.io/Tensorium_lib/";
    license = licenses.mit;
    platforms = platforms.linux ++ platforms.darwin;
  };
}
