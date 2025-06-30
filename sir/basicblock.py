from sir.instruction import Instruction

class BasicBlock:
    def __init__(self, addr_content, pflag):
        # The address of the start of this basic block
        self.addr_content = addr_content
        if addr_content != None:
            # Calculate the integer offset
            self.addr = int(addr_content, base = 16)
        # Instruction list
        self.instructions = []
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

    def HasBranch(self):
        for inst in self.instructions:
            if inst.IsBranch():
                return True
            
        return False

    # Check if the basic block was initialized normally
    def IsInitialized(self):
        return self.addr_content != None

    # Initialize the basic bloci with entry address and branch flag
    def Init(self, addr_content, pflag):
        self.addr_content = addr_content
        self.addr = int(addr_content, base = 16)
        self._PFlag = pflag
    
    # Check if the basic block contains any instructions
    def IsEmpty(self):
        return len(self.instructions) == 0
    
    def GetBranchTarget(self):
        for i in range(len(self.instructions)):
            inst = self.instructions[i]
            if inst.IsBranch():
                if i == len(self.instructions) - 1:
                    # The last instruction in basic block
                    return self.addr + 64
                else:
                    # Not the last instruction in basic block
                    if self.instructions[len(self.instructions) - 1].IsExit():
                        return self.addr + 32
                    
        return 0

    def GetDirectTarget(self, NextBB):
        TargetInst = NextBB.instructions[0]
        if TargetInst.IsExit():
            return NextBB.addr

        return 0

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
        Idx = 0;
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

    # Get the true branch
    def GetTrueBranch(self, Inst):
        # Get the branch flag from branch instruction, i.e. P0 or !P0
        PFlag = Inst.GetBranchFlag()
        if PFlag == None:
            return None

        # Get the basic block that contains branch flag from successors
        for BB in self._succs:
            BPFlag = BB.PFlag
            if PFlag == BPFlag:
                return BB

        return None
    
    # Get the false branch
    def GetFalseBranch(self, Inst):
        # Get the branch flag from branch instruction, i.e. P0 or !P0
        PFlag = Inst.GetBranchFlag()
        if PFlag == None:
            return None
        
        # Get the basic block that does not contain the branch flag from successors
        for BB in self._succs:
            BPFlag = BB.PFlag
            if BPFlag == None:
                return BB

        return None
    
    def Lift(self, lifter, IRBuilder, IRRegs, BlockMap, ConstMem):
        # Handle SSA PHI instructions at the top of the block
        idx = 0
        while idx < len(self.instructions) and self.instructions[idx].opcodes[:1] == ["PHI"]:
            phi_inst = self.instructions[idx]
            # Definition operand is first operand
            def_op = phi_inst.operands[0]
            phi_val = IRBuilder.phi(def_op.GetIRType(lifter), def_op.GetIRRegName(lifter))
            # Incoming operands correspond to predecessors in order
            for i, use_op in enumerate(phi_inst.operands[1:]):
                pred_bb = self._preds[i]
                incoming = IRRegs.get(use_op.GetIRRegName(lifter)) if use_op.IsReg else ConstMem.get(use_op.ArgOffset)
                phi_val.add_incoming(incoming, BlockMap[pred_bb])
            IRRegs[def_op.GetIRRegName(lifter)] = phi_val
            idx += 1


        # Lift remaining instructions until a branch or end
        for inst in self.instructions[idx:]:
            if inst.IsBranch():
                true_br = self.GetTrueBranch(inst)
                false_br = self.GetFalseBranch(inst)
                try:
                    inst.LiftBranch(lifter, IRBuilder, IRRegs, BlockMap[true_br], BlockMap[false_br], ConstMem)
                except Exception as e:
                    lifter.lift_errors.append(e)
                break
            inst.Lift(lifter, IRBuilder, IRRegs, ConstMem)

    def dump(self):
        print("BB Addr: ", self.addr_content)
        for inst in self.instructions:
            inst.dump()
        print("BB End-------------")
        
