nixpkgs: final: prev:
let
  addNativeBuildInputs = drvName: inputs: {
    "${drvName}" = prev.${drvName}.overridePythonAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ inputs;
    });
  };

  addPropagatedBuildInputs = drvName: inputs: {
    "${drvName}" = prev.${drvName}.overridePythonAttrs (old: {
      propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ inputs;
    });
  };

  addBuildInputs = drvName: inputs: {
    "${drvName}" = prev.${drvName}.overridePythonAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ inputs;
    });
  };
in
{
  inherit (nixpkgs.python3Packages) jupyterlab-vim;
  # inherit (nixpkgs.python3Packages) numpy tensorflow matplotlib
    # By prophet
    # pytz six python-dateutil;
}
// addNativeBuildInputs "jupyterlab-vim" [ final.jupyter-packaging ]
# // addNativeBuildInputs "jupyter-client" [ final.hatchling ]
# // addNativeBuildInputs "ipykernel" [ final.hatchling ]
# // addNativeBuildInputs "jupyter-core" [ final.hatchling ]
# // addNativeBuildInputs "comm" [ final.hatchling ]
# // addBuildInputs "pmdarima" [ final.statsmodels ]
# // addBuildInputs "packaging" [ final.flit-core ]
