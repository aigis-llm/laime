{
  pkgs,
  attrs,
  final,
  inputs,
  system,
  lib,
  ...
}:
let
  utils = (
    import ./utils.nix (
      inputs
      // {
        inherit system inputs lib;
      }
    )
  );
in
{
  tests =
    let
      pythonSet = utils.mkPythonSet {
        inherit pkgs;
        withCUDA = false;
        editable = true;
      };
      virtualenv = pythonSet.mkVirtualEnv "laime-testing-env" (utils.getDeps { withCUDA = false; });
    in
    (attrs.tests or { })
    // {
      lint = pkgs.stdenv.mkDerivation {
        name = "${final.laime.name}-lint";
        inherit (final.laime) src;
        nativeBuildInputs = [
          virtualenv
          pkgs.just
        ];
        dontConfigure = true;
        buildPhase = ''
          runHook preBuild
          export REPO_ROOT=$PWD
          ${pkgs.expect}/bin/unbuffer just lint &> out || true
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          mv out $out
          runHook postInstall
        '';
        postInstall = ''
          if [[ $(cat $out | sed $'s/\033\[[0-9;]*m//g') == *"error: Recipe \`lint\` failed on line"* ]]; then
            echo "LINT ERRORS!!! logs:"
            cat $out
            false
          fi
        '';
      };
      test =
        let
          mxbai-embed-large-v1 = pkgs.stdenv.mkDerivation {
            pname = "mxbai-embed-large-v1";
            version = "1.0.0";
            src = ./.;
            nativeBuildInputs = [
              (pkgs.python312.withPackages (ps: [ ps.sentence-transformers ]))
            ];
            buildPhase = ''
              mkdir ./hf-cache
              export HF_HOME="./hf-cache"
              mkdir -p ./hf-cache/hub/models--mixedbread-ai--mxbai-embed-large-v1/refs/
              echo -n "db9d1fe0f31addb4978201b2bf3e577f3f8900d2" > ./hf-cache/hub/models--mixedbread-ai--mxbai-embed-large-v1/refs/main
              python ./download_mxbai_embed_large_v1.py
            '';
            installPhase = ''
              mkdir $out
              mv ./hf-cache/hub/* $out
            '';
            outputHashAlgo = "sha256";
            outputHashMode = "recursive";
            outputHash = "sha256-ptGf339X5W2Cgs/uE1gD1hnVS8S/DPhhTO7AUMR30QU=";
          };
        in
        pkgs.stdenv.mkDerivation {
          name = "${final.laime.name}-test";
          inherit (final.laime) src;
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          nativeBuildInputs = [
            virtualenv
            pkgs.just
          ];
          dontConfigure = true;
          dontBuild = true;
          doCheck = true;
          checkPhase = ''
            runHook preBuild
            mkdir ./hf-cache
            export HF_HOME="./hf-cache"
            cp -r ${mxbai-embed-large-v1} ./hf-cache/hub/
            #ls -la ./hf-cache/hub/*/*/*
            #false
            export HF_HUB_OFFLINE=1
            export REPO_ROOT=$PWD
            ${pkgs.expect}/bin/unbuffer just test &> results || true
            runHook postBuild
          '';
          installPhase = ''
            runHook preInstall
            mkdir $out
            mv coverage.lcov $out
            mv junit.xml $out
            cp results $out
            runHook postInstall
          '';
          postInstall = ''
            if [[ $(cat ./results | sed $'s/\033\[[0-9;]*m//g') == *"error: Recipe \`test\` failed on line"* ]]; then
              echo "TEST ERRORS!!! logs:"
              cat ./results
              false
            fi
          '';
        };
    };
}
