Requirements and Usage

To run this code, you need to install drprobe (available at https://www.er-c.org/barthel/drprobe/
) and drprobe_interface (https://github.com/FWin22/drprobe_interface
), as well as PyTorch, GPyTorch, and BoTorch.

The code has been tested with PyTorch 2.0.1, GPyTorch 1.11, and BoTorch 0.10.0.

To execute the workflow:

Run imaging.py to optimize imaging parameters.

Run position.py to fit atomic column positions and obtain an updated structure.

Run 3Dbuild.py to construct the 3D structure of a single region. In this step, the initial structures were obtained from the optimization of atomic column positions.

Please note that this is an initial release. The full version of the code will be made available after the associated paper is accepted.
