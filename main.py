import argparse
from parse.parser import SaSSParser
from sir.module import Module
from transform.transforms import Transforms
from lift.lifter import Lifter

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='input sass file', dest='input_asm', required=True, metavar='FILE')
  parser.add_argument('-o', '--output', help='output LLVM module name', dest='output_module')
  parser.add_argument('-inc', '--include', help='include files', nargs='+')
  parser.add_argument('-name', '--kernel-name', help='kernel name', dest='kernel_name', default='kern', type=str)
  parser.add_argument('-arch', dest='arch', default=75, type=int, choices=[70, 75, 80, 86])
  args = parser.parse_args()

  # Read input sass file
  with open(args.input_asm, 'r') as input_file:
    file = input_file.read()
    # Skip commands, empty line ...
    # file = ExpandCode(file, args.include)
    # file = ExpandInline(file, args.include)
    # file, regs   = SetRegisterMap(file)
    # file, params = SetParameterMap(file)
    # file, consts = SetConstsMap(file)
    # file   = ReplaceRegParamConstMap(file, regs, params, consts)
    # kernel = assemble(file)

    # Create SaSS parser
    sass_parser = SaSSParser(file)
    
    # Parse file
    m = Module(args.output_module, sass_parser)

    # Apply transformations
    trans = Transforms("SaSS transforms")
    trans.apply(m)
    
    # Setup LLVM
    lifter = Lifter()

    file_name = args.output_module + '.ll'
    with open(file_name, 'w') as output_file:
      # Lift to LLVM IR
      m.Lift(lifter, output_file)
    
if __name__ == '__main__':
  main()
