{ pkgs ? import <nixpkgs> {
    config = { 
      allowUnfree = true; 
    }; 
  } 
}:

pkgs.mkShell {
packages = with pkgs; [
  vscode
  python312Full
  gcc
  clang
  openblas
  openmpi
  valgrind
  cloc
  tree
] ++ (with python312Packages; [
  pip
  virtualenv
  ipykernel
  notebook
  jupyter-client
  pyzmq
  pybind11
  ipykernel

]) ++ (with llvmPackages_18; [
  mlir
  clang
  llvm
  libclang
  
  openmp
]);

  # only for nanobind 
  shellHook = ''
    if [ ! -d .venv ]; then
      echo "[+] Creating .venv..."
      python3 -m venv .venv
      source .venv/bin/activate
      pip install nanobind 
    else
      source .venv/bin/activate
    fi
  '';
}
