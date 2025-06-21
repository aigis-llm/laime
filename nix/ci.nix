{
  lib,
  inputs,
  ...
}:
{
  flake.actions-nix = {
    defaults = {
      jobs = {
        timeout-minutes = 60;
        runs-on = "ubuntu-latest";
      };
    };
    workflows = {
      ".github/workflows/nix-x86_64-linux.yaml" = (
        lib.recursiveUpdate
          (inputs.nix-auto-ci.makeNixGithubAction {
            flake = inputs.self;
            useLix = true;
          })
          {
            jobs.codecov = {
              needs = [ "fast-build" ];
              steps = [
                {
                  uses = "actions/checkout@v4";
                }
                {
                  uses = "actions/download-artifact@v4";
                  "with" = {
                    path = "artifacts";
                  };
                }
                {
                  uses = "codecov/codecov-action@v5";
                  "with" = {
                    token = "\${{ secrets.CODECOV_TOKEN }}";
                    fail_ci_if_error = true;
                    files = "./artifacts/results/result-test/coverage.lcov";
                    disable_search = true;
                  };
                }
                {
                  uses = "codecov/test-results-action@v1";
                  "if" = "\${{ !cancelled() }}";
                  "with" = {
                    token = "\${{ secrets.CODECOV_TOKEN }}";
                    fail_ci_if_error = true;
                    files = "./artifacts/results/result-test/junit.xml";
                    disable_search = true;
                  };
                }
              ];
            };
          }
      );
    };
  };
}
