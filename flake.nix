{
  description = "Description for the project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-parts.inputs.nixpkgs-lib.follows = "nixpkgs";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix-hammer-overrides = {
      url = "github:TyberiusPrime/uv2nix_hammer_overrides";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
    git-hooks.inputs.flake-compat.follows = "";

    actions-nix.url = "github:nialov/actions.nix";
    actions-nix.inputs = {
      nixpkgs.follows = "nixpkgs";
      flake-parts.follows = "flake-parts";
      pre-commit-hooks.follows = "git-hooks";
    };

    nix-auto-ci.url = "github:aigis-llm/nix-auto-ci";
    nix-auto-ci.inputs = {
      nixpkgs.follows = "nixpkgs";
      flake-parts.follows = "flake-parts";
      git-hooks.follows = "git-hooks";
      actions-nix.follows = "actions-nix";
    };
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-parts,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      uv2nix-hammer-overrides,
      git-hooks,
      actions-nix,
      nix-auto-ci,
      self,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {

      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      imports = [
        git-hooks.flakeModule
        actions-nix.flakeModules.default
        nix-auto-ci.flakeModule
      ];

      flake.actions-nix = {
        defaults = {
          jobs = {
            timeout-minutes = 60;
            runs-on = "ubuntu-latest";
          };
        };
        workflows = {
          ".github/workflows/nix-x86_64-linux.yaml" = inputs.nix-auto-ci.makeNixGithubAction {
            flake = self;
            useLix = true;
          };
        };
      };

      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          lib,
          ...
        }:
        let
          inherit
            (import ./nix/utils.nix (
              inputs
              // {
                inherit lib system inputs;
              }
            ))
            mkShell
            mkPythonSet
            mkVirtualEnv
            ;
        in
        {
          devShells.default = mkShell { withCUDA = false; };
          devShells.cuda = mkShell { withCUDA = true; };
          packages.default = mkVirtualEnv "laime-env" { withCUDA = false; };
          packages.cuda = mkVirtualEnv "laime-cuda-env" { withCUDA = true; };
          checks =
            let
              pythonSet = (
                mkPythonSet {
                  withCUDA = false;
                  pkgs = pkgs;
                }
              );
            in
            {
              inherit (pythonSet.laime.passthru.tests) lint test;
            };
        };
    };
}
