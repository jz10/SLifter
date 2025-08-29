from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from lift.lifter import Lifter
from llvmlite import ir
from transform.sr_substitute import SR_TO_OFFSET

class Function:
    
    #def __init__(self, parser):
    #    self.name = ""
    #    self.ParseFunction(parser)

    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.args = []
        self.ArgMap = {}
        self.BlockMap = {}

    # Resovle instructions' operands
    def ResolveOperands(self, insts):
        self.args = []
        self.regs = []
        for inst in insts:
            args, regs = inst.GetArgsAndRegs()
            self.args.extend(args)
            self.regs.extend(regs)
        
    # Get the arguments for current function
    def GetArgs(self):
        ArgIdxes = []
        Args = []

        ArgMap = {}
        # Collect the arguments
        for BB in self.blocks:
            for inst in BB.instructions:
                    for Operand in inst.operands:
                        if Operand.IsArg:
                            Offset = Operand.ArgOffset
                            ArgMap[Offset] = Operand

        # Sort the map
        SortedArgs = {key: val for key, val in
                      sorted(ArgMap.items(), key = lambda ele: ele[0])}

        # Collect the keys
        for Offset, Operand in SortedArgs.items():
            if Operand.Skipped: # This is the case that the operand may be the associated operand
                continue
            else:
                ArgIdxes.append(Offset)
                Args.append(Operand)
                
        return ArgIdxes, Args

    # Get the registers used in this function
    def GetRegs(self, lifter):
        Regs = {}
        for BB in self.blocks:
            BB.GetRegs(Regs, lifter)

        return Regs
        
    # Create control flow graph
    def DumpCFG(self):
        for BB in self.blocks:
            succs = []
            for succ in BB.succs:
                succs.append(succ.addr_content)
            print("BB: ", BB.addr_content, succs)

    # Build the map between basic block and its IR version
    def BuildBBToIRMap(self, IRFunc, BlockMap):
        IsEntry = True
        for BB in self.blocks:
            if IsEntry:
                BBName = "EntryBB_" + BB.addr_content
            else:
                BBName = "BB_" + BB.addr_content

            # Create the basic block
            IRBlock = IRFunc.append_basic_block(BBName)
            # Register IR block
            BlockMap[BB] = IRBlock
            
            IsEntry = False
            
    # Lift to LLVM IR
    def Lift(self, lifter, llvm_module, func_name):
        # Collect arguments
        ArgIdxes, Args = self.GetArgs()
        
        
        FuncTy = lifter.ir.FunctionType(lifter.ir.VoidType(), [])
        IRFunc = lifter.ir.Function(llvm_module, FuncTy, self.name)

        # Construct the map based on IR basic block
        self.BuildBBToIRMap(IRFunc, self.BlockMap)

        # Preload the constant memory values
        ConstMem = {}
        EntryBlock = self.BlockMap[self.blocks[0]]
        Builder = lifter.ir.IRBuilder(EntryBlock)
        Builder.position_at_start(EntryBlock)
        
        OFFSET_TO_SR = {v: k for k, v in SR_TO_OFFSET.items()}

        for entry in Args:
            addr = Builder.gep(lifter.ConstMem, [ir.Constant(ir.IntType(64), 0), 
                                                 ir.Constant(ir.IntType(64), entry.ArgOffset)])
            if entry.ArgOffset in OFFSET_TO_SR:
                name = OFFSET_TO_SR[entry.ArgOffset]
            else:
                name = f"c[0x0][{hex(entry.ArgOffset)}]"

            addr = Builder.bitcast(addr, lifter.ir.PointerType(lifter.GetIRType(entry.TypeDesc)))
            val = Builder.load(addr, name)
            ConstMem[entry.ArgOffset] = val

        # SSA mapping: register names to LLVM IR values
        IRRegs = {}

        # Lower each basic block
        for BB in self.blocks:
            IRBlock = self.BlockMap[BB]
            Builder = lifter.ir.IRBuilder(IRBlock)
            BB.Lift(lifter, Builder, IRRegs, self.BlockMap, ConstMem)

        # Second pass to add incoming phi nodes
        for BB in self.blocks:
            IRBlock = self.BlockMap[BB]
            Builder = lifter.ir.IRBuilder(IRBlock)
            BB.LiftPhiNodes(lifter, Builder, IRRegs, self.BlockMap)

        
