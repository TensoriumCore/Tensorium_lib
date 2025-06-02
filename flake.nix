{
  description = "Tensorium Plugin DevShell (Clang 17 X86 only on macOS)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "x86_64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };

          llvmPkgs =
            if system == "x86_64-darwin" then
              pkgs.llvmPackages_17.override {
                targets = [ "X86" ]; # 💡 sans RISCV
              }
            else
              pkgs.llvmPackages_17;

        in
        {

          default = pkgs.mkShell {
            name = "tensorium-dev";
            packages = [
              llvmPkgs.clang
              llvmPkgs.llvm
              llvmPkgs.libclang
              llvmPkgs.openmp
              pkgs.cmake
              pkgs.python312Packages.pybind11
              pkgs.doxygen
              pkgs.graphviz
              pkgs.bear
              pkgs.cloc
            ];
            shellHook = ''
              echo "✅ Tensorium dev shell (Clang version: $(clang++ --version | head -n 1))"
              which clang++
            '';
          };
        }
      );
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };

          llvmPkgs =
            if system == "x86_64-darwin" then
              pkgs.llvmPackages_17.override {
                targets = [ "X86" ]; # 💡 sans RISCV
              }
            else
              pkgs.llvmPackages_17;

          libTensorium = pkgs.callPackage ./nix {
            inherit llvmPkgs pkgs;
            self = self;
            lib = pkgs.lib;
          };
        in
        {
          default = libTensorium;
          libTensorium = libTensorium;
        }
      );
    };
}
