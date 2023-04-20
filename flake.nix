{
  description = "A package for efficiently working with ternary or binary feature vectors common in phonology.";

  nixConfig.extra-substituters = [
    "https://tweag-jupyter.cachix.org"
  ];
  nixConfig.extra-trusted-public-keys = [
    "tweag-jupyter.cachix.org-1:UtNH4Zs6hVUFpFBTLaA4ejYavPo5EFFqgd7G7FxGW9g="
  ];

  inputs.flake-compat.url = "github:edolstra/flake-compat";
  inputs.flake-compat.flake = false;
  inputs.flake-utils.url = "github:numtide/flake-utils";

  # inputs.nixpkgs.url = "github:nixos/nixpkgs/archive/ec14e43941ac0b49121aa24c49c55eaf89d2f385.tar.gz";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  # inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  inputs.jupyenv.url = "github:tweag/jupyenv";

  outputs = {
    self,
    flake-compat,
    flake-utils,
    nixpkgs,
    jupyenv,
    ...
  } @ inputs:
    flake-utils.lib.eachSystem
    [
      flake-utils.lib.system.x86_64-linux

      # See https://github.com/tweag/jupyenv/issues/388
      # flake-utils.lib.system.aarch64-darwin
    ]
    (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (jupyenv.lib.${system}) mkJupyterlabNew;
        jupyterlab = mkJupyterlabNew ({...}: {
          nixpkgs = inputs.nixpkgs;
          imports = [(import ./kernels.nix)];
        });
      in rec {
        packages = {inherit jupyterlab;};
        packages.default = jupyterlab;

        apps.default.program = "${jupyterlab}/bin/jupyter-notebook";
        # apps.default.program = "${jupyterlab}/bin/jupyter-lab";
        apps.default.type = "app";

        devShell = pkgs.mkShell {
          shellHook = ''
            export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
          '';
          buildInputs = with pkgs; [ poetry ];
          # buildInputs = with pkgs; [ (python310.withPackages (ps: with ps; [ poetry ])) ];
          # buildInputs = with pkgs; [ poetry python-language-server pyright ]
        };
      }
    );
}
