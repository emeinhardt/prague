{pkgs, self, ...}: {
  kernel.python.kitchenSink = {
    enable = true;
    # extraPackages = ps: with ps; [ funcy ];
    projectDir = ./kernels/python;
  };
}
