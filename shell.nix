{ pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  }
}:

pkgs.mkShell {
  buildInputs = with pkgs; [

    vscode
    gcc
    openblas
    openmpi
    valgrind
    cloc
    tree
	doxygen 
	graphviz 
	bear

    python312Full
    (python312.withPackages (ps: with ps; [
      pip
      virtualenv
      ipykernel
      notebook
      jupyter-client
      pyzmq
      pybind11
    ]))

  ] ++ (with llvmPackages_18; [
    mlir
    clang
    llvm
    libclang
    openmp
  ]);

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
