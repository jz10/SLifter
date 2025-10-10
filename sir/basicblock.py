from sir.instruction import Instruction

class BasicBlock:
    def __init__(self, addr_content, pflag, instructions=[]):
        # The address of the start of this basic block
        self.addr_content = addr_content
        if addr_content != None:
            # Calculate the integer offset
            self.addr = int(addr_content, base = 16)
        # Instruction list
        self.instructions = instructions
        # Predecessor
        self._preds = []
        # Successors
        self._succs = []
        # Path flag
        self._PFlag = pflag
        
    @property
    def PFlag(self):
        return self._PFlag
    
    def AppendInst(self, inst):
        self.instructions.append(inst)
        
    def AddPred(self, pred):
        if pred not in self._preds:
            self._preds.append(pred)

    def AddSucc(self, succ):
        if succ not in self._succs:
            if succ == None:
                print("Add none successor???")
            self._succs.append(succ)

    # Check if the basic block was initialized normally
    def IsInitialized(self):
        return self.addr_content != None
    
    def Isolated(self):
        return len(self._preds) == 0 and len(self._succs) == 0

    # Initialize the basic bloci with entry address and branch flag
    def Init(self, addr_content, pflag):
        self.addr_content = addr_content
        self.addr = int(addr_content, base = 16)
        self._PFlag = pflag
    
    # Check if the basic block contains any instructions
    def IsEmpty(self):
        return len(self.instructions) == 0

    # Merge congtent with another basic block
    def Merge(self, another):
        # Append instruction list
        self.instructions = self.instructions + another.instructions
        # Erase old successor
        if another in self._succs:
            self._succs.remove(another)
        # Add new successor
        self._succs = self._succs + another._succs
        
    # Erase the redundency in basic block
    def EraseRedundency(self):
        #inst = self.instructions[0]
        #if inst.IsExit():
        #    # Empty the instruction list and just keep the exit instruction
        #    self.instructions = []
        #    self.instructions.append(inst)
        Idx = 0
        while Idx < len(self.instructions):
            Inst = self.instructions[Idx]
            if Inst.IsNOP():
                # Remove NOP instructions
                self.instructions.remove(Inst)
            elif Inst.IsExit():
                Idx = Idx + 1
                # Erase the rest of instructions
                if Idx < len(self.instructions):
                    del self.instructions[Idx : len(self.instructions)] 
            else:
                Idx = Idx + 1
        
    # Collect registers with type
    def GetRegs(self, Regs, lifter):
        for Inst in self.instructions:
            Inst.GetRegs(Regs, lifter)

    def GetTerminator(self):
        return self.instructions[-1]

    def SetTerminator(self, Inst):
        Inst = Inst.Clone()
        self.instructions[-1] = Inst
        Inst._Parent = self

    def GetBranchPair(self, Inst, Blocks):
        # Get the branch flag from branch instruction, i.e. P0 or !P0
        PFlag = Inst.GetBranchFlag()
        if PFlag == None:
            return None
        
        if len(self._succs) != 2:
            print("Warning: More than/less than two successors in basic block")
            
        BB1 = None
        BB2 = None

        for BB in Blocks:
            BBAddress = BB.addr_content
            if int(BBAddress, 16) == Inst.GetUses()[1].ImmediateValue:
                BB1 = BB
            if int(BBAddress, 16) == Inst.GetUses()[2].ImmediateValue:
                BB2 = BB

        if BB1 == None or BB2 == None:
            print("Warning: Cannot find the branch target basic block")
            return None

        return BB1, BB2

    def Lift(self, lifter, IRBuilder, IRRegs, BlockMap, ConstMem):
        for inst in self.instructions:
            inst.Lift(lifter, IRBuilder, IRRegs, ConstMem, BlockMap)

    def LiftPhiNodes(self, lifter, IRBuilder, IRRegs, BlockMap):
        IRBlock = BlockMap[self]

        def roughSearch(op):
            reg = op.Reg
            name = op.GetIRName(lifter)
            targetType = name.replace(reg, "")

            bestKey = max(IRRegs.keys(), key=lambda k: (k.startswith(reg), len(k)))

            val = IRBuilder.bitcast(IRRegs[bestKey], op.GetIRType(lifter), f"{name}_cast")

            return val

        for i, inst in enumerate(self.instructions):
            if inst.opcodes[0] == "PHI" or inst.opcodes[0] == "PHI64":
                IRInst = IRBlock.instructions[i]

                # Incoming operands correspond to predecessors in order
                for i, op in enumerate(inst._operands[1:]):
                    pred_bb = self._preds[i]
                    if op.IsRZ:
                        val = lifter.ir.Constant(op.GetIRType(lifter), 0)
                    elif op.IsPT:
                        val = lifter.ir.Constant(op.GetIRType(lifter), 1)
                    else:
                        irName = op.GetIRName(lifter)
                        if irName not in IRRegs:
                            val = roughSearch(op)
                        else:
                            val = IRRegs[irName]
                    IRInst.add_incoming(val, BlockMap[pred_bb])

    def dump(self):
        print("BB Addr: ", self.addr_content)
        for inst in self.instructions:
            inst.dump()
        print("BB End-------------")
        
