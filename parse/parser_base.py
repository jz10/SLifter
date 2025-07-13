from sir.function import Function
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.controlcode import ControlCode
from sir.defuse import DefUse
from sir.operand import Operand
from sir.operand import InvalidOperandException
import re

class NoParsingEffort(Exception):
    pass

class UnmatchedControlCode(Exception):
    pass

REG_PREFIX = 'R'
ARG_PREFIX = 'c[0x0]'
ARG_OFFSET = 320 # 0x140

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

        # Code rearrange: if line ISETP Pn is not followed by @Pn, move the line until it is
        lines = self.rearrange_isetp_lines(lines)

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
            # Create def-use chain
            CurrFunc.DefUse = self.CreateDefUse(Insts)
            
            # Wrap up previous function by creating control-flow graph
            CurrFunc.blocks = self.CreateCFG(self.SplitBlocks(Insts))

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
                    CurrFunc.blocks = self.CreateCFG(self.SplitBlocks(Insts))
                    
                # Check the function name
                items = line.split(' : ')

                # Create new function
                return Function(items[1])

        return None

    # Parse the function body from file lines
    def ParseFuncBody(self, line, Insts):
        raise NoParsingEffort

    # Split basic block from the list instruction
    def SplitBlocks(self, Insts):
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
        if PFlag != None:
            Opcodes.insert(0, PFlag)

        # Parse operands
        Operands = []
        for Operand_Content in Operands_Content:
            Operands.append(self.ParseOperand(Operand_Content, CurrFunc))

        # Create instruction
        return Instruction(InstID, Opcodes, Operands, Opcode_Content + " " + Operands_Detail)

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
        if Operand_Content.startswith('0x'):
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
        if Operand_Content.find(PATH_PREFIX) == 0: # operand starts from 'P'
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
   
    # Build the control-flow graph
    def BuildCFG(self, Blocks):
        # No need to process single basic block case
        if len(Blocks) == 1:
            return Blocks

        # Handle the multi-block case
        JumpTargets = {}
        for i in range(len(Blocks)):
            CurrBB = Blocks[i]
            # Handle branch target case: the branch instruciton locates in current basic block
            self.CheckAndAddTarget(CurrBB, CurrBB.GetBranchTarget(), JumpTargets)
 
            # Handle the direct target case: the next basic block contains exit instruction
            if i < len(Blocks) - 1:
                self.CheckAndAddTarget(CurrBB, CurrBB.GetDirectTarget(Blocks[i + 1]), JumpTargets)

        MergedTo = {}
        NewBlocks = []
        CurrBB = None
        for i in range(len(Blocks)):
            NextBB = Blocks[i]
            # Add CFG connection to its jump source
            if NextBB.addr in JumpTargets:
                for TargetBB in JumpTargets[NextBB.addr]:
                    if TargetBB in MergedTo:
                        TargetBB = MergedTo[TargetBB]
                    
                    TargetBB.AddSucc(NextBB)
                    NextBB.AddPred(TargetBB)
                # Reset current basic block, i.e. restart potential merging
                CurrBB = None
                
            # Handle the basic block merge case
            if CurrBB != None and NextBB.addr not in JumpTargets:
                # Merge two basic blocks
                CurrBB.Merge(NextBB)
                MergedTo[NextBB] = CurrBB

                continue
            
            if CurrBB == None:
                # Reset current basic block
                CurrBB = NextBB
                # Add current basic block to translated block list
                NewBlocks.append(CurrBB)

        for NewBB in NewBlocks:
            NewBB.EraseRedundency()
            # NewBB.dump()
                
        return NewBlocks

    #Create the control-flow graph
    def CreateCFG(self, Blocks):
        # No need to process single basic block case
        if len(Blocks) == 1:
            return Blocks

        for BB in Blocks:
            BB.EraseRedundency()
            # BB.dump()

        DefUse.BuildGlobalDU(Blocks)
            
        return Blocks
    
    # Check if the target address is legal, then add the target address associated with its jump source
    def CheckAndAddTarget(self, CurrBB, TargetAddr, JumpTargets):
        if TargetAddr > 0:
            if TargetAddr not in JumpTargets:
                JumpTargets[TargetAddr] = []
            JumpTargets[TargetAddr].append(CurrBB)

    # Create def-use chain
    def CreateDefUse(self, Insts):
        DU = DefUse()

        # The table of current definition
        CurrDefs = {}
        
        # Iterate through instructions
        #for Inst in Insts:
            # Check the use and connect the defs
            # Get def and put on current defs table
            #Inst.controlcode.dump()
    def rearrange_isetp_lines(self, lines):
        isetp_pattern = r'/\*[0-9a-fA-F]+\*/\s+ISETP\.[A-Z.]+\s+([P]\d+),'
        predicate_usage_pattern = r'/\*[0-9a-fA-F]+\*/\s+@([P]\d+|![P]\d+)\s+'
        
        rearranged_lines = []
        isetp_instructions = {}
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            isetp_match = re.search(isetp_pattern, line)
            if isetp_match:
                predicate = isetp_match.group(1)

                isetp_instructions[predicate] = (i, line)
                i += 1
                continue
            
            predicate_match = re.search(predicate_usage_pattern, line)
            if predicate_match:
                used_predicate = predicate_match.group(1)
                if used_predicate.startswith('!'):
                    used_predicate = used_predicate[1:]
                
                if used_predicate in isetp_instructions:
                    stored_line_index, stored_line = isetp_instructions[used_predicate]
                    rearranged_lines.append(stored_line)
                    del isetp_instructions[used_predicate]
            
            rearranged_lines.append(line)
            i += 1
        
        return rearranged_lines
