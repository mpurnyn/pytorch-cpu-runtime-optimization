# ML Task

This is a short interview coding task at Coram.AI for the machine learning candidates.

The task tests whether the candidates are familiar with pytorch and libtorch.

## Setup

1. Install [docker](https://docs.docker.com/engine/install/). Recommended version is `23.0.3`.
2. Run the code using `./run_stack.sh`. Underhood, this command builds the docker image, launches the container, and executes the binaries for your test.
   - This setup has been tested on both Ubuntu and MacOS with M1 chips.
   - This setup has been tested on Windows 11 6r bit Asus G14 with a AMD Ryzen 9 5900HS with Radeon Graphics Chip.
     - `./run.ps1` may be executed on Windows

## Tasks

1. Finish the code in `src/model_generation_main.py` to complete the 3 tasks defined in the TODOs in the source code.
2. Finish the code in `src/pytorch_main.cc` to complete the 2 tasks defined in the TODOs in the source code.

You are free to edit the code in any way you want to finish the tasks above.

## Tips

To quickly iterate on your code, you can:

1. manually log into the container
2. iterate on your code outside the container
3. test your command by manually executing `build_and_execute.sh`

Refer to `run_stack.sh` for more full details.
