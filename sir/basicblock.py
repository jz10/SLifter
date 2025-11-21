class BasicBlock:
    def __init__(self, addr_content, pflag, instructions=[]):
        # The address of the start of this basic block
        self.addr_content = addr_content
        if addr_content != None:
            self.addr_content = addr_content.lower()
            # Calculate the integer offset
            self.addr = int(addr_content, base = 16)
        # Instruction list
        self.instructions = instructions
        # Predecessor
        self.preds = []
        # Successors
        self.succs = []
        # Path flag
        self.pflag = pflag
        
    def add_pred(self, pred):
        if pred not in self.preds:
            self.preds.append(pred)

    def add_succ(self, succ):
        if succ not in self.succs:
            if succ == None:
                print("Add none successor???")
            self.succs.append(succ)
    
    # def is_isolated(self):
    #     return len(self.preds) == 0 and len(self.succs) == 0

    # # Initialize the basic bloci with entry address and branch flag
    # def init_block(self, addr_content, pflag):
    #     self.addr_content = addr_content
    #     self.addr = int(addr_content, base = 16)
    #     self.pflag = pflag
    
    # # Check if the basic block contains any instructions
    # def is_empty(self):
    #     return len(self.instructions) == 0

    # # Merge congtent with another basic block
    # def merge(self, another):
    #     # Append instruction list
    #     self.instructions = self.instructions + another.instructions
    #     # Erase old successor
    #     if another in self.succs:
    #         self.succs.remove(another)
    #     # Add new successor
    #     self.succs = self.succs + another.succs
        
    # # Erase the redundency in basic block
    # def erase_redundancy(self):
    #     #inst = self.instructions[0]
    #     #if inst.is_exit():
    #     #    # Empty the instruction list and just keep the exit instruction
    #     #    self.instructions = []
    #     #    self.instructions.append(inst)
    #     Idx = 0
    #     while Idx < len(self.instructions):
    #         Inst = self.instructions[Idx]
    #         if Inst.is_nop():
    #             # Remove NOP instructions
    #             self.instructions.remove(Inst)
    #         elif Inst.is_exit():
    #             Idx = Idx + 1
    #             # Erase the rest of instructions
    #             if Idx < len(self.instructions):
    #                 del self.instructions[Idx : len(self.instructions)] 
    #         else:
    #             Idx = Idx + 1
        
    # Collect registers with type
    def get_regs(self, Regs, lifter):
        for Inst in self.instructions:
            Inst.get_regs(Regs, lifter)

    def get_terminator(self):
        LastInst = self.instructions[-1]
        if LastInst.is_exit() or LastInst.is_branch() or LastInst.is_return():
            return LastInst
        else:
            return None

    def set_terminator(self, Inst):
        Inst = Inst.clone()
        self.instructions[-1] = Inst
        Inst.parent = self

    def get_branch_pair(self, Inst, Blocks):
        # Get the branch flag from branch instruction, i.e. P0 or !P0
        PFlag = Inst.get_branch_flag()
        if PFlag == None:
            return None
        
        if len(self.succs) != 2:
            print("Warning: More than/less than two successors in basic block")
            
        BB1 = None
        BB2 = None

        for BB in Blocks:
            BBAddress = BB.addr_content
            if int(BBAddress, 16) == Inst.get_uses()[1].immediate_value:
                BB1 = BB
            if int(BBAddress, 16) == Inst.get_uses()[2].immediate_value:
                BB2 = BB

        if BB1 == None or BB2 == None:
            print("Warning: Cannot find the branch target basic block")
            return None

        return BB1, BB2

    def dump(self):
        print("BB Addr: ", self.addr_content)
        for inst in self.instructions:
            inst.dump()
        print("BB End-------------")
        
