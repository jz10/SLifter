#!/usr/bin/env python3
"""
NVBit to SASS Converter
Converts NVBit trace output to SASS-like format
"""

import re
import sys
from typing import List, Dict, Tuple

class Kernel:
    def __init__(self, name: str, mangled_name: str):
        self.name = name
        self.mangled_name = mangled_name
        self.instructions: List[Tuple[int, str]] = []
        self.num_thread_blocks = 0
        self.kernel_instructions = 0
        self.total_instructions = 0

class NVBitToSASSConverter:
    def __init__(self):
        self.kernels: List[Kernel] = []
        self.total_app_instructions = 0
        
    def parse_nvbit_file(self, filepath: str):
        """Parse NVBit trace file and extract kernel information"""
        current_kernel = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Parse kernel inspection start
            if line.startswith("inspecting"):
                match = re.match(r"inspecting\s+(.+?)\s+-\s+num instrs\s+(\d+)", line)
                if match:
                    kernel_name = match.group(1)
                    # Extract clean function name (remove template parameters for display)
                    clean_name = re.sub(r'<[^>]+>', '', kernel_name)
                    current_kernel = Kernel(clean_name, kernel_name)
                    
            # Parse instruction lines
            elif line.startswith("Instr") and current_kernel:
                match = re.match(r"Instr\s+(\d+)\s+@\s+(0x[0-9a-fA-F]+)\s+\((\d+)\)\s+-\s+(.+?)\s*;", line)
                if match:
                    instr_num = int(match.group(1))
                    offset = int(match.group(2), 16)
                    instruction = match.group(4)
                    current_kernel.instructions.append((offset, instruction))
            
            # Parse kernel summary
            elif line.startswith("kernel") and current_kernel:
                match = re.match(r"kernel\s+\d+\s+-\s+(\S+)\s+-\s+#thread-blocks\s+(\d+),\s+kernel instructions\s+(\d+),\s+total instructions\s+(\d+)", line)
                if match:
                    current_kernel.mangled_name = match.group(1)
                    current_kernel.num_thread_blocks = int(match.group(2))
                    current_kernel.kernel_instructions = int(match.group(3))
                    current_kernel.total_instructions = int(match.group(4))
                    self.kernels.append(current_kernel)
                    current_kernel = None
            
            # Parse total app instructions
            elif line.startswith("Total app instructions:"):
                match = re.match(r"Total app instructions:\s+(\d+)", line)
                if match:
                    self.total_app_instructions = int(match.group(1))
            
            i += 1
    
    def generate_sass_output(self) -> str:
        """Generate SASS-formatted output"""
        output = []
        
        # Add Fatbin header (using sample values as requested)
        output.append("Fatbin elf code:")
        output.append("================")
        output.append("arch = sm_52")
        output.append("code version = [1,7]")
        output.append("host = linux")
        output.append("compile_size = 64bit")
        output.append("")
        output.append("\tcode for sm_52")
        output.append("")
        
        # Process each kernel
        for kernel in self.kernels:
            output.append("Fatbin elf code:")
            output.append("================")
            output.append("arch = sm_52")
            output.append("code version = [1,7]")
            output.append("host = linux")
            output.append("compile_size = 64bit")
            output.append("")
            output.append("\tcode for sm_52")
            output.append(f"\t\tFunction : {kernel.mangled_name}")
            output.append('\t.headerflags    @"EF_CUDA_SM52 EF_CUDA_PTX_SM(EF_CUDA_SM52)"')
            
            # Generate control codes (placeholder values)
            output.append(" " * 73 + "/* 0x001cfc00e22007f6 */")
            
            # Process instructions
            for i, (offset, instruction) in enumerate(kernel.instructions):
                # Format offset as 4-digit hex with leading zeros
                offset_str = f"{offset:04x}"
                
                # Add control code line every 4-8 instructions (similar to SASS format)
                if i > 0 and i % 4 == 0:
                    output.append(" " * 73 + "/* 0x001fd842fec20ff1 */")
                
                # Format instruction line
                # SASS format: /*offset*/ INSTRUCTION ; /* hex_encoding */
                instr_line = f"        /*{offset_str.upper()}*/                   {instruction} ;"
                
                # Add placeholder hex encoding
                hex_encoding = f"0x{i:016x}"
                instr_line = f"{instr_line:<73} /* {hex_encoding} */"
                
                output.append(instr_line)
            
            output.append("\t\t..........")
            output.append("")
            output.append("")
            output.append("")
        
        # Add PTX section (placeholder)
        output.append("Fatbin ptx code:")
        output.append("================")
        output.append("arch = sm_52")
        output.append("code version = [7,8]")
        output.append("host = linux")
        output.append("compile_size = 64bit")
        output.append("compressed")
        output.append("")
        
        return "\n".join(output)
    
    def convert(self, input_file: str, output_file: str):
        """Main conversion function"""
        print(f"Parsing NVBit trace file: {input_file}")
        self.parse_nvbit_file(input_file)
        
        print(f"Found {len(self.kernels)} kernels")
        print(f"Total app instructions: {self.total_app_instructions}")
        
        sass_output = self.generate_sass_output()
        
        with open(output_file, 'w') as f:
            f.write(sass_output)
        
        print(f"SASS output written to: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python nvbit_to_sass.py <input_nvbit_file> <output_sass_file>")
        print("Example: python nvbit_to_sass.py trace.txt output.sass")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    converter = NVBitToSASSConverter()
    
    try:
        converter.convert(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()