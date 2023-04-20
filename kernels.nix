{pkgs, self, ...}: {
  kernel.python.kitchen-sink = {
    enable = true;
    # extraPackages = [  ]
    projectDir = ./kernels/python;
  };
}
