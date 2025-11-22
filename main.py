import argparse
import pathlib
from parse.parser import SaSSParser
from sir.module import Module
from lift.lifter_nvvm import NVVMLifter
from lift.lifter_x86 import X86Lifter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_asm",
        required=True,
        nargs="+",
        metavar="FILE",
    )
    parser.add_argument(
        "-o", "--output", help="output LLVM module name", dest="output_module"
    )
    parser.add_argument(
        "-name",
        "--kernel-name",
        help="kernel name",
        dest="kernel_name",
        default="kern",
        type=str,
    )
    parser.add_argument(
        "-arch", dest="arch", default=75, type=int, choices=[70, 75, 80, 86]
    )
    parser.add_argument(
        "--lifter",
        choices=["nvvm", "x86"],
        default="nvvm",
        help="select lifter backend",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="emit lifter stdout logging"
    )
    args = parser.parse_args()

    with open(args.input_asm[0], "r") as input_sass:
        file = input_sass.read()
        sass_parser = SaSSParser(file)
        module = Module(args.output_module, sass_parser)
        
    input_elf = None
    if len(args.input_asm) > 1:
        elf_file_path = args.input_asm[1]
        p = pathlib.Path(elf_file_path)
        if p.exists():
            input_elf = p.read_text()
        

    verbose_mode = bool(args.verbose)
    if args.lifter == "nvvm":
        lifter = NVVMLifter(elf=input_elf, verbose=verbose_mode)
    elif args.lifter == "x86":
        lifter = X86Lifter(elf=input_elf, verbose=verbose_mode)
    else:
        raise ValueError(f"Unknown lifter backend '{args.lifter}'")
    
        

    file_name = args.output_module + ".ll"
    with open(file_name, "w") as output_file:
        lifter.lift_module(module, output_file)


if __name__ == "__main__":
    main()
