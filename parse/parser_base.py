from sir.function import Function
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.controlcode import ControlCode
from sir.operand import Operand
from sir.operand import InvalidOperandException
import re

class NoParsingEffort(Exception):
    pass

class UnmatchedControlCode(Exception):
    pass

REG_PREFIX = 'R'
UREG_PREFIX = 'UR' 
ARG_PREFIX = 'c[0x0]'

# Special register constants  
SR_TID = 'SR_TID'
SR_NTID = 'SR_NTID'
SR_CTAID = 'SR_CTAID'
SR_LANE = 'SR_LANE'
SR_WARP = 'SR_WARP'
SR_ZERO = 'SRZ'

PTRREG_PREFIX = '[R'
PATH_PREFIX = 'P'
PTR_PREFIX = '['
PTR_SUFFIX = ']'

class SaSSParserBase:
    def __init__(self, isa, file):
        self.file = file

    # Parse the SaSS text file
    def apply(self):
        # List of functions
        Funcs = []

        # The current function that is under parsing
        CurrFunc = None

        # The instruction list
        Insts = []

        # Handle dual issue(remove curly braces)
        lines = self.file.split('\n')
        modified_lines = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if '{' in line:
                line = line.replace("{", "")
            if "}" in line:
                if line[-1] != "/":
                    line = line + lines[i + 1]
                    line = line.replace("\n", "")
                    skip_next = True
                line = line.replace("}",";")
            modified_lines.append(line)

        lines = modified_lines

        # Main loop that parse the SaSS text file
        for line_num, line in enumerate(lines):
            # Process lines in SaSS text file

            # Process function title and misc
            PrevFunc = CurrFunc
            CurrFunc = self.CreateFunction(line, PrevFunc, Insts)
            if PrevFunc != CurrFunc and CurrFunc != None:
                # Just create a new functino
                if PrevFunc != None:
                    # Add the parsed function
                    Funcs.append(PrevFunc)
                    # Clean the instruction list
                    Insts = []
                    
                continue
            else:
                CurrFunc = PrevFunc
                
            # Process function body
            self.ParseFuncBody(line, Insts, CurrFunc)

        # Process the last function
        if CurrFunc != None:
            # Wrap up previous function by creating control-flow graph
            CurrFunc.blocks = self.CreateCFG(Insts)

            # Add the parsed function
            Funcs.append(CurrFunc)
            
        return Funcs

    # Parse the file line to create new function
    def CreateFunction(self, line, PrevFunc, Insts):
        if (not ("/*" in line and "*/" in line)):
            # Check function start
            if ("Function : " in line):
                # Wrap up previous function by creating control-flow graph
                if PrevFunc != None:
                    PrevFunc.blocks = self.CreateCFG(Insts)
                # Check the function name
                items = line.split(' : ')

                # Create new function
                return Function(items[1])

        return None

    # Parse the function body from file lines
    def ParseFuncBody(self, line, Insts):
        raise NoParsingEffort



    # Retrieve instruction ID
    def GetInstNum(self, line):
        items = line.split('/*')
        return items[1]
    
    def ExtractPredicateReg(self, opcode):
        if opcode[0] == '@' and opcode[1] == 'P' and opcode[2].isdigit():
            return opcode[1:3]
        
        return None
    
    def ExtractPredicateRegNeg(self, opcode):
        if opcode[0] == '@' and opcode[1] == '!' and opcode[2] == 'P' and opcode[3].isdigit():
            return opcode[1:4]
            
        return None
    
    # Retrieve instruction's opcode
    def GetInstOpcode(self, line):
        items = line.split(';')
        line = (items[0].lstrip())
        items = line.split(' ')
        # Get opcode
        opcode = items[0]
        PFlag = None
        # Handle the condntion branch flags
        # if opcode == "@P0":
        #     PFlag = "P0"
        #     opcode = items[1]
        # elif opcode == "@!P0":
        #     PFlag = "!P0"
        #     opcode = items[1]
        pred_reg = self.ExtractPredicateReg(opcode)
        pred_reg_neg = self.ExtractPredicateRegNeg(opcode)
        rest_content = line.replace(items[0], "")
        if pred_reg:
            PFlag = pred_reg 
            opcode = items[1]
            rest_content = rest_content.replace(items[1], "")
        elif pred_reg_neg:
            PFlag = pred_reg_neg
            opcode = items[1]
            rest_content = rest_content.replace(items[1], "")
            
        return opcode, PFlag, rest_content

    # Retrieve instruction's operands
    def GetInstOperands(self, line):
        items = line.split(',')
        ops = []
        for item in items:
            operand = item.lstrip()
            if (operand != ''):
                ops.append(operand)

        return ops 

    # Parse instruction, includes operators and operands
    def ParseInstruction(self, InstID, Opcode_Content, PFlag, Operands_Content, Operands_Detail, CurrFunc):
        # Parse opcodes
        Opcodes = Opcode_Content.split('.')

        # Parse operands
        Operands = []
        for Operand_Content in Operands_Content:
            Operands.append(self.ParseOperand(Operand_Content, CurrFunc))

        # Create instruction
        return Instruction(InstID, Opcodes, Operands, Opcode_Content + " " + Operands_Detail, None, PFlag)

    # Parse operand
    def ParseOperand(self, Operand_Content, CurrFunc):
        Operand_Content = Operand_Content.lstrip()
        IsReg = False
        Reg = None
        Name = None
        Suffix = None
        ArgOffset = -1
        IsArg = False
        IsMemAddr = False
        IsImmediate = False
        ImmediateValue = None
        # Check if it is an immediate value
        if Operand_Content.startswith('0x') or Operand_Content.startswith('-0x'):
            IsImmediate = True
            Operand_Content = Operand_Content.replace('0x', '')
            # Convert to integer
            ImmediateValue = int(Operand_Content, base = 16)
            Name = Operand_Content
            Reg = None
            Suffix = None
            ArgOffset = -1  # Reset ArgOffset since this is not an argument
            
            return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate, ImmediateValue)

        # Check if it is a register for address pointer, e.g. [R0]
        if Operand_Content.find(PTRREG_PREFIX) == 0: # operand starts from '[R'
            # Fill out the ptr related charactors
            Operand_Content = Operand_Content.replace(PTR_PREFIX, "")
            Operand_Content = Operand_Content.replace(PTR_SUFFIX, "")
            IsMemAddr = True
            
        # Check if it is a register (including -, ~, or abs '|' as prefix)
        if Operand_Content.startswith(REG_PREFIX) or Operand_Content.startswith('-') or Operand_Content.startswith('~') or Operand_Content.startswith('|'):
            IsReg = True
            Reg = Operand_Content
            Name = Operand_Content
            
            # Get the suffix of given operand
            items = Reg.split('.')
            if len(items) > 1:
                Reg = items[0]
                Suffix = items[1]
                if len(items) > 2 and items[2] != "reuse": # reuse is a assembly hint for register reuse
                        raise InvalidOperandException
                
            return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate, None)

        # Check if it is a jump flag
        if Operand_Content.find(PATH_PREFIX) == 0 or Operand_Content.startswith('!'): # operand starts from 'P' or '!
            IsReg = True
            Reg = Operand_Content
            Name = Operand_Content

            return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate, None)
        
        # Check if it is a function argument or dimension related value
        if Operand_Content.find(ARG_PREFIX) == 0: 
            IsArg = True

            # Get the suffix of given operand
            items = Operand_Content.split('.')
            if len(items) > 1:
                Suffix = items[1]
                if len(items) > 2:
                    raise InvalidOperandException

                Operand_Content = items[0]
        
            ArgOffset = self.GetArgOffset(Operand_Content.replace(ARG_PREFIX, ""))
            Name = Operand_Content
            # Create argument operaand
            Arg = Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate)

            return Arg
        
        # Special zero register
        if Operand_Content == SR_ZERO:
            IsReg = True
            Reg = Operand_Content
            Name = Operand_Content
            Suffix = None
            
            return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate, None)
        
        # Check if it is a special register value
        items = Operand_Content.split('.')
        if len(items) >= 1:
            special_regs = [SR_TID, SR_NTID, SR_CTAID, SR_LANE, SR_WARP]
            for special_reg in special_regs:
                if items[0].find(special_reg) == 0:
                    Name = Operand_Content
                    if len(items) > 1:
                        Suffix = items[1]
                    break

        return Operand(Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate, None)

    # Parse the argument offset
    def GetArgOffset(self, offset):
        offset = offset.replace('[', "")
        offset = offset.replace(']', "")
        return int(offset, base = 16)

    # Parse control code 
    def ParseControlCode(self, Content, ControlCodes):
       raise NoParsingEffort

    #Create the control-flow graph
    def CreateCFG(self, Insts):
        # Preprocess 1: branch may target non-instruction address
        # Align address upward to the next instruction
        addr_set  = {int(inst.id,16) for inst in Insts} 
        addr_list = sorted(a for a in addr_set)
        def _align_up_to_inst(addr_hex):
            addr = int(addr_hex, 16)
            max_addr = addr_list[-1]

            while addr not in addr_set and addr <= max_addr:
                addr += 0x8 
            return addr
        for inst in Insts:
            if inst.IsBranch():
                target_addr = _align_up_to_inst(inst.operands[0].Name.zfill(4))
                inst.operands[0]._Name = format(target_addr, '04x')
                inst.operands[0]._ImmediateValue = target_addr

        # Preprocess 2: insert NOP so predicated instructions are not adjacent
        NewInsts = []
        InstIdCounter = int(Insts[-1].id, 16)
        for i, inst in enumerate(Insts):
            if i > 0 and inst.Predicated() and Insts[i-1].Predicated():
                # Insert NOP instruction
                nop_inst = Instruction(
                    id=f"{hex(InstIdCounter + 1).zfill(4)}",
                    opcodes=["NOP"],
                    operands=[],
                    inst_content="NOP",
                    parentBB=None,
                    pflag=None
                )
                NewInsts.append(nop_inst)
            NewInsts.append(inst)
        Insts = NewInsts

        Blocks = []

        # Identify leaders
        leaders = set()
        pflags = {}
        for i, inst in enumerate(Insts):
            if inst.IsReturn() or inst.IsExit():
                if i + 1 < len(Insts):
                    leaders.add(Insts[i+1].id)
            if inst.IsBranch():
                leaders.add(inst.operands[0].Name.zfill(4))
            if inst.Predicated():
                leaders.add(Insts[i].id)
                pflags[Insts[i].id] = inst.pflag
                if i + 1 < len(Insts):
                    leaders.add(Insts[i+1].id)

        # Create basic blocks
        BlockInsts = []
        pflag = None
        for inst in Insts:
            if inst.id in leaders:
                if BlockInsts:
                    # Make sure every block contains a terminator
                    if not BlockInsts or (not BlockInsts[-1].IsExit() and not BlockInsts[-1].IsReturn() and not BlockInsts[-1].IsBranch()):
                        DestOp = Operand(inst.id, None, None, -1, False, False, False, True, int(inst.id, 16))
                        NewInst = Instruction(
                            id=f"branch_{int(inst.id, 16)}",
                            opcodes=["BRA"],
                            operands=[DestOp],
                            inst_content=f"BRA {DestOp.Name}",
                            parentBB=None,
                            pflag=None
                        )
                        BlockInsts.append(NewInst)

                    Block = BasicBlock(BlockInsts[0].id, pflag, BlockInsts)
                    pflag = pflags.get(inst.id, None)
                    Blocks.append(Block)
                BlockInsts = [inst]
            else:
                BlockInsts.append(inst)

        # Preprocessor and sucessors
        idToBlock = {block.addr_content: block for block in Blocks}

        for i, block in enumerate(Blocks):
            last = block.instructions[-1]
            first = block.instructions[0]

            # Add fallthrough if predicated
            if first.Predicated():
                if i + 1 < len(Blocks) and i - 1 >= 0:
                    nextBlock = Blocks[i + 1]
                    prevBlock = Blocks[i - 1]
                    prevBlock.AddSucc(nextBlock)
                    nextBlock.AddPred(prevBlock)

            # If the last instruction is a return or exit, no successors
            if last.IsReturn() or last.IsExit():
                continue

            elif last.IsBranch():
                targetBlock = idToBlock.get(last.operands[0].Name.zfill(4))
                if targetBlock is None:
                    continue

                block.AddSucc(targetBlock)
                targetBlock.AddPred(block)
            
            else:
                if i + 1 < len(Blocks):
                    nextBlock = Blocks[i + 1]
                    block.AddSucc(nextBlock)
                    nextBlock.AddPred(block)


        # Add conditional branch for the predecessor of the predicated instruction
        for block in Blocks:
            firstInst = block.instructions[0]
            if firstInst.Predicated():
                
                op = Operand(firstInst.pflag, firstInst.pflag, None, -1, True, False, False)
                NewInst = Instruction(
                    id=f"pbra_{firstInst.id}",
                    opcodes=["PBRA"],
                    operands=[op],
                    inst_content=f"PBRA {firstInst.pflag}",
                    parentBB=None,
                    pflag=None
                )

                if len(block._preds) != 1:
                    print(f"Warning: predicate block predecessor != 1")

                predBlock = block._preds[0]
                predBlock.SetTerminator(NewInst)

        # Set parent for each instruction
        for block in Blocks:
            for inst in block.instructions:
                inst._Parent = block
                
        # No need to process single basic block case
        if len(Blocks) == 1:
            return Blocks

        for BB in Blocks:
            BB.EraseRedundency() # Remove NOP and dead instructions after exit(spin-loop senteniel)
            if BB.Isolated():
                Blocks.remove(BB)
            
        return Blocks
    
    # Check if the target address is legal, then add the target address associated with its jump source
    def CheckAndAddTarget(self, CurrBB, TargetAddr, JumpTargets):
        if TargetAddr > 0:
            if TargetAddr not in JumpTargets:
                JumpTargets[TargetAddr] = []
            JumpTargets[TargetAddr].append(CurrBB)