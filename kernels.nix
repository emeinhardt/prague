{pkgs, self, ...}: {
  kernel.python.kitchen-sink = {
    enable = true;
    # extraPackages = [  ]
    # projectDir = ./.;
    # projectDir = ./kernels/python;
    # projectDir = "./kernels/python";
    # projectDir = self + modules/kernels/python;
    projectDir = self + kernels/python;
    # projectDir = self + "kernels/python";
  };
}
