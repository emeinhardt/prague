{pkgs, ...}: {
  kernel.python.kitchen-sink = {
    enable = true;
    # extraPackages = [  ]
    # projectDir = self + "/kernels/python";
  };
}
