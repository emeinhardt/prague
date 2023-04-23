{pkgs, self, ...}:
  {
  kernel.python.kitchenSink = {
    enable = true;
    # extraPackages = ps: with ps; [ funcy ];
    projectDir = ./kernels/python;
    overrides = (import ./overrides.nix pkgs);
    # overrides = pkgs.poetry2nix.overrides.withDefaults (import ./overrides.nix pkgs);
    # overrides = pkgs.poetry2nix.overrides.withDefaults (
    #   final: prev:
    #   let
    #     addNativeBuildInputs = drvName: inputs: {
    #       "${drvName}" = prev.${drvName}.overridePythonAttrs (old: {
    #         nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ inputs;
    #       });
    #     };

    #     addPropagatedBuildInputs = drvName: inputs: {
    #       "${drvName}" = prev.${drvName}.overridePythonAttrs (old: {
    #         propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ inputs;
    #       });
    #     };

    #     addBuildInputs = drvName: inputs: {
    #       "${drvName}" = prev.${drvName}.overridePythonAttrs (old: {
    #         buildInputs = (old.buildInputs or [ ]) ++ inputs;
    #       });
    #     };
    #   in
    #   {
    #     inherit (pkgs.python3Packages) jupyterlab-vim;
    #   }
    #   // addNativeBuildInputs "jupyterlab-vim" [ final.jupyter-packaging ]
    #   # // addNativeBuildInputs "jupyter-client" [ final.hatchling ]
    #   # // addNativeBuildInputs "ipykernel" [ final.hatchling ]
    #   # // addNativeBuildInputs "jupyter-core" [ final.hatchling ]
    #   # // addNativeBuildInputs "comm" [ final.hatchling ]
    #   # // addBuildInputs "pmdarima" [ final.statsmodels ]
    #   # // addBuildInputs "packaging" [ final.flit-core ]
    # );
  };
}

# {pkgs, self, ...}: {
#   kernel.python.kitchenSink = {
#     enable = true;
#     # extraPackages = ps: with ps; [ funcy ];
#     projectDir = ./kernels/python;
#     # overrides = ./overrides.nix;
#     # overrides = pkgs.poetry2nix.overrides.withDefaults (import ./overrides.nix pkgs);
#   };
# }
