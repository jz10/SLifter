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
    
    # Retrieve instruction's opcode
    def GetInstOpcode(self, line):
        items = line.split(';')
        line = (items[0].lstrip())
        items = line.split(' ')
        # Get opcode
        opcode = items[0]
        PFlag = None

        pred_reg = None
        not_ = False
        if opcode.startswith('@'):
            opcode = opcode[1:]        
        if opcode.startswith('!'):
            not_ = True
            opcode = opcode[1:]
        if opcode.startswith('P') and opcode[1].isdigit():
            pred_reg = opcode
        
        rest_content = line.replace(items[0], "")
        
        if pred_reg:
            PFlag = Operand.fromReg(pred_reg, pred_reg, None, False, not_, False)
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
            Operands.append(Operand.Parse(Operand_Content))

        # Create instruction
        return Instruction(InstID, Opcodes, Operands, Opcode_Content + " " + Operands_Detail, None, PFlag)

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
                target_addr = _align_up_to_inst(inst.operands[-1].Name.zfill(4))
                inst.operands[0]._Name = format(target_addr, '04x')
                inst.operands[0]._ImmediateValue = target_addr

        # Preprocess 2: insert NOP so predicated instructions are not adjacent
        NewInsts = []
        InstIdCounter = int(Insts[-1].id, 16)
        for i, inst in enumerate(Insts):
            if i > 0 and inst.Predicated() and Insts[i-1].Predicated():
                # Insert NOP instruction
                nop_inst = Instruction(
                    id=f"{hex(InstIdCounter + 1)[2:].zfill(4)}",
                    opcodes=["NOP"],
                    operands=[],
                    inst_content="NOP",
                    parentBB=None,
                    pflag=None
                )
                NewInsts.append(nop_inst)
                InstIdCounter += 1
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
                        DestOp = Operand.fromImmediate(inst.id, int(inst.id, 16))                        
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

        # print("Predicate converted CFG:")
        # for block in Blocks:
        #     print(f"  Block: {block.addr_content}", end="")
        #     print(f" from: [", end="")
        #     for pred in block._preds:
        #         print(f"{pred.addr_content},", end="")
        #     print(f"]", end="")
        #     print(f" to: [", end="")
        #     for succ in block._succs:
        #         print(f"{succ.addr_content},", end="")
        #     print(f"]")
        #     for inst in block.instructions:
        #         print(f"    {inst.id}    {inst}")


        # Add conditional branch for the predecessor of the predicated instruction
        for i, block in enumerate(Blocks):
            firstInst = block.instructions[0]
            if firstInst.Predicated():                
                op = firstInst.pflag.Clone()

                if len(block._succs) > 1:
                    print(f"Warning: predicated block successor > 1")
                    raise UnmatchedControlCode

                TrueBrBlock = block
                FalseBrBlock = Blocks[i + 1]

                srcOp1 = Operand.fromImmediate(TrueBrBlock.addr_content, int(TrueBrBlock.addr_content, 16))
                srcOp2 = Operand.fromImmediate(FalseBrBlock.addr_content, int(FalseBrBlock.addr_content, 16))

                NewInst = Instruction(
                    id=f"pbra_{firstInst.id}",
                    opcodes=["PBRA"],
                    operands=[op, srcOp1, srcOp2],
                    parentBB=None,
                    pflag=None
                )
                
                # Remove predication from the original instruction
                firstInst._PFlag = None

                for pred in block._preds:
                    if FalseBrBlock not in pred._succs:
                        pred.AddSucc(FalseBrBlock)
                    if pred not in FalseBrBlock._preds:
                        FalseBrBlock.AddPred(pred)
                    pred.SetTerminator(NewInst)

        # print("Add conditional branch for the predecessor of the predicated instruction:")
        # for block in Blocks:
        #     print(f"  Block: {block.addr_content}", end="")
        #     print(f" from: [", end="")
        #     for pred in block._preds:
        #         print(f"{pred.addr_content},", end="")
        #     print(f"]", end="")
        #     print(f" to: [", end="")
        #     for succ in block._succs:
        #         print(f"{succ.addr_content},", end="")
        #     print(f"]")
        #     for inst in block.instructions:
        #         print(f"    {inst.id}    {inst}")

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