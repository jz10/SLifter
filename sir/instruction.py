from sir.operand import Operand
from sir.controlcode import ControlCode
from sir.controlcode import PresetCtlCodeException

class Instruction:
    def __init__(self, id, opcodes, operands, parentBB=None, pflag=None):
        self.id = id
        self.opcodes = opcodes
        self.operands = operands
        self.inst_content = None
        self.true_branch = None
        self.false_branch = None
        self.ctl_code = None
        self.parent = parentBB
        self.pflag = pflag

        self.users = {}
        self.reaching_defs = {}

        # Def/use operands layout correction
        self.use_op_start_idx = 1

        if len(self.opcodes) > 0 and (self.opcodes[0] == "PHI" or self.opcodes[0] == "PHI64"):
            return

        # IMAD.WIDE has two defs, RN+1:RN
        if len(self.opcodes) > 1 and self.opcodes[0] == "IMAD" and self.opcodes[1] == "WIDE":
            reg_pair_name = 'R' + str(int(self.operands[0].reg[1:]) + 1)
            reg_pair = Operand.from_reg(reg_pair_name, reg_pair_name)
            self.operands.insert(1, reg_pair)
            self.use_op_start_idx = 2
        
        # UIMAD.WIDE has two defs, RN+1:RN
        if len(self.opcodes) > 1 and self.opcodes[0] == "UIMAD" and self.opcodes[1] == "WIDE":
            reg_pair_name = 'UR' + str(int(self.operands[0].reg[2:]) + 1)
            reg_pair = Operand.from_reg(reg_pair_name, reg_pair_name)
            self.operands.insert(1, reg_pair)
            self.use_op_start_idx = 2
            
        # BAR.SYNC 0x0
        elif len(self.opcodes) > 1 and self.opcodes[0] == "BAR":
            self.use_op_start_idx = 0
            
        # SHFL.DOWN PT, R59 = R18, 0x8, 0x1f
        elif len(self.opcodes) > 1 and self.opcodes[0] == "SHFL":
            self.use_op_start_idx = 2
            
        # ISETP.LT.U32.OR = P0, PT, R12, R10, P0;
        elif "SETP" in self.opcodes[0]:
            self.use_op_start_idx = 2

        # LDG.E.64.SYS R4 = [R2] defines R4 and R5
        elif self.is_load() and "64" in self.opcodes and len(self.operands) > 1:
            reg_pair_name = None
            if "UR" in self.operands[0].reg:
                reg_pair_name = 'UR' + str(int(self.operands[0].reg[2:]) + 1)
            else:
                reg_pair_name = 'R' + str(int(self.operands[0].reg[1:]) + 1)
            reg_pair = Operand.from_reg(reg_pair_name, reg_pair_name)
            self.operands.insert(1, reg_pair)
            self.use_op_start_idx = 2
            
        # LOP3.LUT R9, RZ =  R21, RZ, 0x33, !PT ; or
        # LOP3.LUT R6 = RZ, R4, RZ, 0x33, !PT ; 
        elif "LOP3" in self.opcodes[0] and self.opcodes[1] == "LUT":
            if self.get_defs()[0].is_predicate_reg:
                self.use_op_start_idx = 2
            else:
                self.use_op_start_idx = 1

        elif len(self.opcodes) > 1 and self.opcodes[0] == "UIMAD" and self.opcodes[1] == "WIDE":
            reg_pair_name = 'UR' + str(int(self.operands[0].reg[2:]) + 1)
            reg_pair = Operand.from_reg(reg_pair_name, reg_pair_name)
            self.operands.insert(1, reg_pair)
            self.use_op_start_idx = 2

        elif len(self.opcodes) > 1 and self.opcodes[0] == "HMMA":
            # HMMA.1688.F32 R20, R38, R57, R20
            # HMMA.<shape>.<accum> D, A, B, C
            # A/D R20 => R20, R21, R22, R23
            # B R38 => R38, R39
            # C R57 => R57, R58
            DReg0 = self.operands[0]
            DReg1 = DReg0.clone()
            DReg1.set_reg('R' + str(int(DReg0.reg[1:]) + 1))
            DReg2 = DReg0.clone()
            DReg2.set_reg('R' + str(int(DReg0.reg[1:]) + 2))
            DReg3 = DReg0.clone()
            DReg3.set_reg('R' + str(int(DReg0.reg[1:]) + 3))

            AReg0 = self.operands[1]
            AReg1 = AReg0.clone()
            AReg1.set_reg('R' + str(int(AReg0.reg[1:]) + 1))
            BReg0 = self.operands[2]
            BReg1 = BReg0.clone()
            BReg1.set_reg('R' + str(int(BReg0.reg[1:]) + 1))

            CReg0 = self.operands[3]
            CReg1 = CReg0.clone()
            CReg1.set_reg('R' + str(int(CReg0.reg[1:]) + 1))
            CReg2 = CReg0.clone()
            CReg2.set_reg('R' + str(int(CReg0.reg[1:]) + 2))
            CReg3 = CReg0.clone()
            CReg3.set_reg('R' + str(int(CReg0.reg[1:]) + 3))

            self.operands = [DReg0, DReg1, DReg2, DReg3, AReg0, AReg1, BReg0, BReg1, CReg0, CReg1, CReg2, CReg3]
            self.use_op_start_idx = 4
            
        # Store and Branch have no def op
        elif self.is_branch() or self.is_store() or self.opcodes[0] == "RED":
            self.use_op_start_idx = 0
        # instruction with predicate carry out have two def op
        elif len(self.operands) > 1 and self.operands[0].is_reg and self.operands[1].is_predicate_reg:
            i = 1
            while i < len(self.operands) and self.operands[i].is_predicate_reg:
                i += 1
            self.use_op_start_idx = i
        elif self.opcodes[0] == "UNPACK64":
            self.use_op_start_idx = 2
            
        # temp solution: add parent field to operand here
        for op in self.operands:
            # if hasattr(op, 'Parent'):
            #     raise Exception("Operand parent already set")
            op.Parent = self

    def clone(self):
        cloned_operands = [op.clone() for op in self.operands]
        cloned_pflag = self.pflag.clone() if self.pflag else None
        cloned_inst = Instruction(
            id=self.id,
            opcodes=self.opcodes.copy(),
            operands=cloned_operands,
            parentBB=self.parent,
            pflag=cloned_pflag
        )
        cloned_inst.inst_content = self.inst_content
        cloned_inst.ctl_code = self.ctl_code
        return cloned_inst

    def set_ctl_code(self, CtlCode):
        if self.ctl_code != None:
            raise PresetCtlCodeException

        self.ctl_code = CtlCode
        
    def is_exit(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "EXIT"

    def is_branch(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] == "BRA" or self.opcodes[0] == "PBRA")
    
    def is_return(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "RET"
    
    def is_set_predicate(self):
        return self.opcodes[0] == "ISETP"
    
    def is_phi(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] == "PHI" or self.opcodes[0] == "PHI64")
        
    def predicated(self):
        return self.pflag is not None

    def is_conditional_branch(self):
        return self.is_branch() and self.predicated()

    def in_cond_path(self):
        return self.predicated()
    
    def is_nop(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "NOP"

    def is_addr_compute(self):
        if len(self.opcodes) > 0 and self.opcodes[0] == "IADD":
            # Check operands
            if len(self.operands) == 3:
                operand = self.operands[2]
            # Check function argument operand
                return operand.is_arg

        return False

    def is_load(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["LDG", "LD", "LDS", "LDC", "ULDC", "LDL"])
    
    def is_global_load(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["LDG"])

    def is_store(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["STG", "SUST", "ST", "STS", "STL"])
    
    def is_global_store(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["STG"])
           
    # Collect registers used in instructions
    def get_regs(self, Regs, lifter):
        for operand in self.operands:
            if operand.is_reg:
                if operand.type_desc == "NOTYPE":
                    print("Warning: Operand type is NOTYPE: ", operand.name)
                Regs[operand.get_ir_name(lifter)] = Operand

    def get_reg_name(self, Reg):
        return Reg.split('@')[0]
    
    def rename_reg(self, Reg, Inst):
        reg_name = self.get_reg_name(Reg)
        new_reg = reg_name + "@" + str(Inst.id)
        return new_reg
    
    # Get def operand
    def get_defs(self):
        return self.operands[:self.use_op_start_idx]
    
    def get_def_by_reg(self, Reg):
        for defOp in self.get_defs():
            if defOp.reg == Reg:
                return defOp
        return None

    # Get use operand
    def get_uses(self):
        return self.operands[self.use_op_start_idx:]
    
    def get_uses_with_predicate(self):
        uses = self.get_uses()
        if self.predicated():
            uses = [self.pflag] + uses
        return uses
    
    # Get branch flag
    def get_branch_flag(self):
        operand = self.operands[0]
        if self.is_predicate_reg(operand.reg):
            return operand.name
        else:
            return None
    
    def __str__(self):
        pred_prefix = f"@{self.pflag} " if self.pflag else ""
        opcodes_str  = '.'.join(self.opcodes)
        def_strs  = [str(op) for op in self.get_defs()]
        use_strs  = [str(op) for op in self.get_uses()]
        operand_section = ""

        if def_strs:
            operand_section += ", ".join(def_strs)
            if use_strs:
                operand_section += " = "

        if use_strs:
            operand_section += ", ".join(use_strs)

        if operand_section:
            return f"{pred_prefix}{opcodes_str} {operand_section}"
        else:
            return f"{pred_prefix}{opcodes_str}"
    
    def __repr__(self):
        return '<' + self.__str__() + '>'

    def is_predicate_reg(self, opcode):
        if opcode[0] == 'P' and opcode[1].isdigit():
            return True
        if opcode[0] == '!' and opcode[1] == 'P' and opcode[2].isdigit():
            return True
        return False

    def dump(self):
        print("inst: ", self.id, self.opcodes)
        for operand in self.operands:
            operand.dump()
