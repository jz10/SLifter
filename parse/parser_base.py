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

        # Identify leaders
        leaders = set()
        predicated_leaders = set()
        curr_pred = None
        for i, inst in enumerate(Insts):
            if inst.IsReturn() or inst.IsExit():
                if i + 1 < len(Insts):
                    leaders.add(Insts[i+1].id)
            if inst.IsBranch():
                leaders.add(inst.operands[0].Name.zfill(4))
            if curr_pred != inst.pflag:
                leaders.add(Insts[i].id)
                if inst.Predicated():
                    predicated_leaders.add(Insts[i].id)
                curr_pred = inst.pflag

        # Create basic blocks
        Blocks = []
        BlockInsts = []
        pflag = None
        PrevBlock = None
        NextBlock = {}
        BlockByAddr = {}
        PredicatedBlocks = set()
        for inst in Insts:
            if inst.id in leaders:
                if BlockInsts:
                    BlockId = f"{int(BlockInsts[0].id, 16):04X}"
                    
                    # If leader in predicated leaders, insert a PBRA at start,
                    # remove the predicate of other instructions
                    PredicatedBlock = False
                    if BlockInsts[0].id in predicated_leaders:
                        PredicatedBlock = True
                        BlockInsts.insert(0, Instruction(
                            id=f"{int(BlockInsts[0].id, 16):04X}",
                            opcodes=["PBRA"],
                            operands=[BlockInsts[0].pflag.Clone()],
                            inst_content=f"PBRA {BlockInsts[0].pflag.Name}",
                            parentBB=None,
                            pflag=None
                        ))
                        BlockInsts[1]._id = f"{int(BlockInsts[0].id, 16)+1:04X}"
                        for PredInst in BlockInsts:
                            PredInst._PFlag = None

                    # Create block
                    Block = BasicBlock(BlockId, pflag, BlockInsts)
                    Blocks.append(Block)
                    
                    if PrevBlock:
                        NextBlock[PrevBlock] = Block
                    PrevBlock = Block
                    
                    BlockByAddr[Block.addr_content] = Block
                    
                    if PredicatedBlock:
                        PredicatedBlocks.add(Block)
                    
                BlockInsts = [inst]
            else:
                BlockInsts.append(inst)
                
        # Construct CFG edges
        for Block in Blocks:
            Terminator = Block.GetTerminator()
            
            # If no branch, insert branch to the next block
            if not Terminator:
                NextB = NextBlock[Block]
                DestOp = Operand.fromImmediate(NextB.addr_content, int(NextB.addr_content, 16))                        
                NewInst = Instruction(
                    id=f"{int(Block.instructions[-1].id, 16)+1:04X}",
                    opcodes=["BRA"],
                    operands=[DestOp]
                )
                Block.instructions.append(NewInst)
                
                Block.AddSucc(NextB)
                NextB.AddPred(Block)
            elif Terminator.IsBranch():
                # If the terminator is a branch, add the successor
                DestOp = Terminator.GetUses()[0]
                NextB = BlockByAddr[DestOp.Name]
                Block.AddSucc(NextB)
                NextB.AddPred(Block)
            elif Terminator.IsReturn() or Terminator.IsExit():
                # If the terminator is a return or exit, do nothing
                pass
            else:
                raise Exception("Unrecognized terminator")
            
        # Process predicated blocks
        InsertBlock = {}
        for block in PredicatedBlocks:
            # Split block by leaving pbra in the original block, rest put in a new block
            pbraInst = block.instructions[0]
            NewBlockInsts = block.instructions[1:]
            NewBlock = BasicBlock(NewBlockInsts[0].id, None, NewBlockInsts)
            block.instructions = [pbraInst]
            
            InsertBlock[block] = NewBlock
            BlockByAddr[NewBlock.addr_content] = NewBlock
            
            # Keep predecessor to first block, move sucessors to new block
            NewBlock._succs = block._succs
            block._succs = []
            for succ in NewBlock._succs:
                succ._preds.remove(block)
                succ.AddPred(NewBlock)
            
            NewBlock.AddPred(block)
            block.AddSucc(NewBlock)
            
            block.AddSucc(NextBlock[block])
            NextBlock[block].AddPred(block)
            
            # Update pbra instruction to point to the two blocks
            TrueBrBlock = NewBlock
            FalseBrBlock = NextBlock[block]
            pbraInst._operands.append(Operand.fromImmediate(TrueBrBlock.addr_content, int(TrueBrBlock.addr_content, 16)))
            pbraInst._operands.append(Operand.fromImmediate(FalseBrBlock.addr_content, int(FalseBrBlock.addr_content, 16)))
            
        OldBlocks = Blocks.copy()
        Blocks = []
        for block in OldBlocks:
            Blocks.append(block)
            if block in InsertBlock:
                Blocks.append(InsertBlock[block])


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

        # Set parent for each instruction
        for block in Blocks:
            for inst in block.instructions:
                inst._Parent = block
                
        # # No need to process single basic block case
        # if len(Blocks) == 1:
        #     return Blocks

        # for BB in Blocks:
        #     BB.EraseRedundency() # Remove NOP and dead instructions after exit(spin-loop senteniel)
        #     if BB.Isolated():
        #         Blocks.remove(BB)
            
        return Blocks
    
    # Check if the target address is legal, then add the target address associated with its jump source
    def CheckAndAddTarget(self, CurrBB, TargetAddr, JumpTargets):
        if TargetAddr > 0:
            if TargetAddr not in JumpTargets:
                JumpTargets[TargetAddr] = []
            JumpTargets[TargetAddr].append(CurrBB)