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
    def GetArgs(self, lifter):
        Args = []

        ArgMap = {}
        # Collect the arguments
        for BB in self.blocks:
            for inst in BB.instructions:
                for Operand in inst.operands:
                    if Operand.IsArg:
                        Name = Operand.GetIRName(lifter)
                        ArgMap[Name] = Operand

        # Sort the map
        Args = sorted(ArgMap.values(), key=lambda x: x.ArgOffset)

        return Args

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
