{
  nixpkgs,
  flake-parts,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  uv2nix-hammer-overrides,
  lib,
  system,
  ...
}:
let
  getDeps =
    { withCUDA }:
    let
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };
    in
    workspace.deps.default
    // {
      laime = [
        (if (withCUDA) then "cuda" else "cpu")
        "dev"
      ];
    };
  mkPythonSet =
    {
      withCUDA,
      pkgs,
      editable ? false,
    }:
    (
      let
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };
        deps = getDeps { withCUDA = withCUDA; };
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
          dependencies = deps;
        };
        pyprojectOverrides =
          (import ./pyproject-overrides.nix { inherit uv2nix-hammer-overrides pkgs withCUDA; })
          .pyprojectOverrides;
        editableOverlay = workspace.mkEditablePyprojectOverlay {
          # Use environment variable
          root = "$REPO_ROOT";
        };
        python = pkgs.python312; # TODO: python313 does not work
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
            stdenv = pkgs.stdenv.override {
              targetPlatform = pkgs.stdenv.targetPlatform // {
                # should fix macos eval failing to find wheel
                darwinSdkVersion = "12.7";
              };
            };
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );

        editablePythonSet = pythonSet.overrideScope editableOverlay;
      in
      (if (editable) then editablePythonSet else pythonSet)
    );
  getNixpkgs =
    { withCUDA }:
    import nixpkgs {
      inherit system;
      config.allowUnfreePredicate =
        pkg:
        builtins.elem (lib.getName pkg) (
          if (withCUDA) then
            [
              "cuda_cccl"
              "cuda_cudart"
              "libcublas"
              "cuda_nvcc"
              "cuda_cupti"
              "cuda_nvrtc"
              "cudnn"
              "libcufft"
              "libcurand"
              "libcusolver"
              "libnvjitlink"
              "libcusparse"
              "libcusparse_lt"
              "libcufile"
            ]
          else
            [ ]
        );
    };
  mkVirtualEnv =
    name:
    {
      withCUDA,
      editable ? false,
    }:
    (mkPythonSet {
      withCUDA = withCUDA;
      pkgs = getNixpkgs { withCUDA = withCUDA; };
      inherit editable;
    }).mkVirtualEnv
      name
      (getDeps {
        withCUDA = withCUDA;
      });
  mkShell =
    {
      withCUDA,
    }:
    let
      pkgs = getNixpkgs { withCUDA = withCUDA; };
    in
    (pkgs.mkShell {
      packages = [
        pkgs.nixfmt-rfc-style
        pkgs.just
        (mkVirtualEnv "laime-dev-env" {
          withCUDA = withCUDA;
          editable = true;
        })
        pkgs.uv
      ];

      shellHook = ''
        # Undo dependency propagation by nixpkgs.
        unset PYTHONPATH
        # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
        export REPO_ROOT=$(git rev-parse --show-toplevel)
        # Make uv use our Python.
        export UV_PYTHON=$(which python)
        # Stop uv from syncing
        export UV_NO_SYNC=1
        # Stop uv from downloading python
        export UV_PYTHON_DOWNLOADS=never
        # llama.cpp
        export LLAMA_CPP_LIB_PATH=${pkgs.llama-cpp.override { cudaSupport = withCUDA; }}/lib
      '';
    });
in
{
  inherit mkShell mkVirtualEnv mkPythonSet;
}
