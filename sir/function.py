class Function:
    
    #def __init__(self, parser):
    #    self.name = ""
    #    self.ParseFunction(parser)

    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.args = []
        self.arg_map = {}
        self.block_map = {}

    # Resovle instructions' operands
    def resolve_operands(self, insts):
        self.args = []
        self.regs = []
        for inst in insts:
            args, regs = inst.get_args_and_regs()
            self.args.extend(args)
            self.regs.extend(regs)
        
    # Get the arguments for current function
    def get_args(self, lifter):
        args = []
        arg_map = {}
        # Collect the arguments
        for bb in self.blocks:
            for inst in bb.instructions:
                for operand in inst.operands:
                    if operand.is_arg:
                        name = operand.get_ir_name(lifter)
                        arg_map[name] = operand

        # Sort the map
        args = sorted(arg_map.values(), key=lambda x: x.arg_offset)

        return args

    # Get the registers used in this function
    def get_regs(self, lifter):
        regs = {}
        for bb in self.blocks:
            bb.get_regs(regs, lifter)

        return regs
        
    # Create control flow graph
    def dump_cfg(self):
        for bb in self.blocks:
            succs = []
            for succ in bb.succs:
                succs.append(succ.addr_content)
            print("BB: ", bb.addr_content, succs)

    # Build the map between basic block and its IR version
    def build_bb_to_ir_map(self, ir_func, block_map):
        is_entry = True
        for bb in self.blocks:
            if is_entry:
                bb_name = "EntryBB_" + bb.addr_content
            else:
                bb_name = "BB_" + bb.addr_content

            # Create the basic block
            ir_block = ir_func.append_basic_block(bb_name)
            # Register IR block
            block_map[bb] = ir_block
            
            is_entry = False
