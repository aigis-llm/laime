{
  pkgs,
  uv2nix-hammer-overrides,
  withCUDA,
  ...
}:
{
  pyprojectOverrides = pkgs.lib.composeExtensions (uv2nix-hammer-overrides.overrides_strict pkgs) (
    final: prev: {
      hatchling = prev.hatchling.overrideAttrs (attrs: {
        propagatedBuildInputs = attrs.propagatedBuildInputs or [ ] ++ [ final.editables ];
      });
      llama-cpp-python = prev.llama-cpp-python.overrideAttrs (attrs: {
        CMAKE_ARGS = [ "-DLLAMA_BUILD=OFF" ];
        propagatedBuildInputs = attrs.propagatedBuildInputs or [ ] ++ [
          pkgs.cmake
          pkgs.git
        ];
        nativeBuildInputs =
          attrs.nativeBuildInputs or [ ]
          ++ (final.resolveBuildSystem {
            scikit-build-core = [ ];
          });
      });
      torch = prev.torch.overrideAttrs (attrs: {
        buildInputs =
          if (withCUDA) then
            [
              pkgs.cudaPackages.cuda_cudart
              pkgs.cudaPackages.cuda_cupti
              pkgs.cudaPackages.cuda_nvrtc
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.libcufft
              pkgs.cudaPackages.libcurand
              pkgs.cudaPackages.libcusolver
              pkgs.cudaPackages.libcusparse
              pkgs.cudaPackages.nccl
              pkgs.cudaPackages.cusparselt
              pkgs.cudaPackages.libcufile
            ]
          else
            [ ];
        #propagatedBuildInputs =
        #  attrs.propagatedBuildInputs or [ ]
        #  ++ (if (withCUDA) then [ pkgs.cudaPackages.cusparselt ] else [ ]);
      });
      laime = prev.laime.overrideAttrs (attrs: {
        passthru = attrs.passthru // (import ./tests.nix { inherit pkgs attrs final; });
      });
    }
  );
}
