from sir.operand import Operand
from sir.controlcode import ControlCode
from sir.controlcode import PresetCtlCodeException

class Instruction:
    def __init__(self, id, opcodes, operands, parentBB=None, pflag=None):
        self._id = id
        self._opcodes = opcodes
        self._operands = operands
        self._InstContent = None
        self._TrueBranch = None
        self._FalseBranch = None
        self._CtlCode = None
        self._Parent = parentBB
        self._PFlag = pflag

        self.Users = {}
        self.ReachingDefs = {}

        # Def/use operands layout correction
        self._UseOpStartIdx = 1

        if len(self._opcodes) > 0 and (self._opcodes[0] == "PHI" or self._opcodes[0] == "PHI64"):
            return

        # IMAD.WIDE has two defs, RN+1:RN
        if len(self.opcodes) > 1 and self.opcodes[0] == "IMAD" and self.opcodes[1] == "WIDE":
            RegPair = self._operands[0].Clone()
            RegPair.SetReg('R' + str(int(RegPair.Reg[1:]) + 1))
            self._operands.insert(1, RegPair)
            self._UseOpStartIdx = 2
            
        # SHFL.DOWN PT, R59 = R18, 0x8, 0x1f
        elif len(self.opcodes) > 1 and self.opcodes[0] == "SHFL":
            self._UseOpStartIdx = 2

        # LDG.E.64.SYS R4 = [R2] defines R4 and R5
        elif self.IsLoad() and "64" in self.opcodes and len(self._operands) > 1:
            RegPair = self._operands[0].Clone()
            if "UR" in RegPair.Reg:
                RegPair.SetReg('UR' + str(int(RegPair.Reg[2:]) + 1))
            else:
                RegPair.SetReg('R' + str(int(RegPair.Reg[1:]) + 1))
            self._operands.insert(1, RegPair)
            self._UseOpStartIdx = 2

        elif len(self.opcodes) > 1 and self.opcodes[0] == "UIMAD" and self.opcodes[1] == "WIDE":
            RegPair = self._operands[0].Clone()
            RegPair.SetReg('UR' + str(int(RegPair.Reg[2:]) + 1))
            self._operands.insert(1, RegPair)
            self._UseOpStartIdx = 2

        elif len(self.opcodes) > 1 and self.opcodes[0] == "HMMA":
            # HMMA.1688.F32 R20, R38, R57, R20
            # HMMA.<shape>.<accum> D, A, B, C
            # A/D R20 => R20, R21, R22, R23
            # B R38 => R38, R39
            # C R57 => R57, R58
            DReg0 = self._operands[0]
            DReg1 = DReg0.Clone()
            DReg1.SetReg('R' + str(int(DReg0.Reg[1:]) + 1))
            DReg2 = DReg0.Clone()
            DReg2.SetReg('R' + str(int(DReg0.Reg[1:]) + 2))
            DReg3 = DReg0.Clone()
            DReg3.SetReg('R' + str(int(DReg0.Reg[1:]) + 3))

            AReg0 = self._operands[1]
            AReg1 = AReg0.Clone()
            AReg1.SetReg('R' + str(int(AReg0.Reg[1:]) + 1))
            BReg0 = self._operands[2]
            BReg1 = BReg0.Clone()
            BReg1.SetReg('R' + str(int(BReg0.Reg[1:]) + 1))

            CReg0 = self._operands[3]
            CReg1 = CReg0.Clone()
            CReg1.SetReg('R' + str(int(CReg0.Reg[1:]) + 1))
            CReg2 = CReg0.Clone()
            CReg2.SetReg('R' + str(int(CReg0.Reg[1:]) + 2))
            CReg3 = CReg0.Clone()
            CReg3.SetReg('R' + str(int(CReg0.Reg[1:]) + 3))

            self._operands = [DReg0, DReg1, DReg2, DReg3, AReg0, AReg1, BReg0, BReg1, CReg0, CReg1, CReg2, CReg3]
            self._UseOpStartIdx = 4
            
        # Store and Branch have no def op
        elif self.IsBranch() or self.IsStore() or self.opcodes[0] == "RED":
            self._UseOpStartIdx = 0
        # instruction with predicate carry out have two def op
        elif len(self._operands) > 1 and self._operands[0].IsReg and self._operands[1].IsPredicateReg:
            i = 1
            while i < len(self._operands) and self._operands[i].IsPredicateReg:
                i += 1
            self._UseOpStartIdx = i
        elif self.opcodes[0] == "UNPACK64":
            self._UseOpStartIdx = 2
            
        # temp solution: add parent field to operand here
        for op in self._operands:
            # if hasattr(op, 'Parent'):
            #     raise Exception("Operand parent already set")
            op.Parent = self

        
    @property
    def id(self):
        return self._id
    
    @property
    def opcodes(self):
        return self._opcodes

    @property
    def operands(self):
        return self._operands

    @property
    def parent(self):
        return self._Parent

    @property
    def pflag(self):
        return self._PFlag

    @property
    def controlcode(self):
        return self._CtlCode


    @property
    def useOpStartIdx(self):
        return self._UseOpStartIdx
    
    def Clone(self):
        cloned_operands = [op.Clone() for op in self._operands]
        cloned_pflag = self._PFlag.Clone() if self._PFlag else None
        cloned_inst = Instruction(
            id=self._id,
            opcodes=self._opcodes.copy(),
            operands=cloned_operands,
            parentBB=self._Parent,
            pflag=cloned_pflag
        )
        cloned_inst._InstContent = self._InstContent
        cloned_inst._CtlCode = self._CtlCode
        return cloned_inst
    
    def ReachingDefsFor(self, str):
        for useOp, defInsts in self.ReachingDefsSet.items():
            if useOp.Reg == str:
                return defInsts
        return None
    
    def GetArgsAndRegs(self):
        regs = []
        args = []
        for operand in self._operands:
            if operand.IsReg:
                regs.append(operand)
            if operand.IsArg:
                args.append(operand)

        return args, regs

    def SetCtlCode(self, CtlCode):
        if self._CtlCode != None:
            raise PresetCtlCodeException

        self._CtlCode = CtlCode
        
    def IsExit(self):
        return len(self._opcodes) > 0 and self._opcodes[0] == "EXIT"

    def IsBranch(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] == "BRA" or self._opcodes[0] == "PBRA")
    
    def IsReturn(self):
        return len(self._opcodes) > 0 and self._opcodes[0] == "RET"
    
    def IsSetPredicate(self):
        return self._opcodes[0] == "ISETP"
    
    def IsPhi(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] == "PHI" or self._opcodes[0] == "PHI64")
        
    def Predicated(self):
        return self._PFlag is not None

    def IsConditionalBranch(self):
        return self.IsBranch() and self.Predicated()

    def InCondPath(self):
        return self.Predicated()
    
    def IsNOP(self):
        return len(self._opcodes) > 0 and self._opcodes[0] == "NOP"

    def IsAddrCompute(self):
        if len(self._opcodes) > 0 and self._opcodes[0] == "IADD":
            # Check operands
            if len(self._operands) == 3:
                operand = self._operands[2]
            # Check function argument operand
                return operand.IsArg

        return False

    def IsLoad(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["LDG", "LD", "LDS", "LDC", "ULDC", "LDL"])
    
    def IsGlobalLoad(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["LDG"])

    def IsStore(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["STG", "SUST", "ST", "STS", "STL"])
    
    def IsGlobalStore(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["STG"])

    # Set all operands as skipped
    def SetSkip(self):
        for Operand in self._operands:
            Operand.SetSkip()
           
    # Collect registers used in instructions
    def GetRegs(self, Regs, lifter):
        for Operand in self._operands:
            if Operand.IsReg:
                if Operand.TypeDesc == "NOTYPE":
                    print("Warning: Operand type is NOTYPE: ", Operand.Name)
                Regs[Operand.GetIRName(lifter)] = Operand

    def GetRegName(self, Reg):
        return Reg.split('@')[0]
    
    def RenameReg(self, Reg, Inst):
        RegName = self.GetRegName(Reg)
        NewReg = RegName + "@" + str(Inst.id)
        return NewReg
    
    # Get def operand
    def GetDefs(self):
        return self._operands[:self._UseOpStartIdx]
    
    def GetDef(self):
        defs = self.GetDefs()
        
        if len(defs) > 1:
            print("Warning: GetDef finds more than one def operand")

        return defs[0] if len(defs) > 0 else None
    
    def GetDefByReg(self, Reg):
        for defOp in self.GetDefs():
            if defOp.Reg == Reg:
                return defOp
        return None

    # Get use operand
    def GetUses(self):
        return self._operands[self._UseOpStartIdx:]
    
    def GetUsesWithPredicate(self):
        uses = self.GetUses()
        if self.Predicated():
            uses = [self.pflag] + uses
        return uses
    
    # Get branch flag
    def GetBranchFlag(self):
        Operand = self._operands[0]
        if self.IsPredicateReg(Operand.Reg):
            return Operand.Name
        else:
            return None
    
    def __str__(self):
        pred_prefix = f"@{self._PFlag} " if self._PFlag else ""
        opcodes_str  = '.'.join(self._opcodes)
        def_strs  = [str(op) for op in self.GetDefs()]
        use_strs  = [str(op) for op in self.GetUses()]
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

    def IsPredicateReg(self, opcode):
        if opcode[0] == 'P' and opcode[1].isdigit():
            return True
        if opcode[0] == '!' and opcode[1] == 'P' and opcode[2].isdigit():
            return True
        return False
    
    def ParseInstructionModifiers(self, opcodes):
        modifiers = {}
        for opcode in opcodes:
            if opcode.startswith('.'):
                if opcode in ['.S32', '.U32', '.S16', '.U16', '.S64', '.U64']:
                    modifiers['type'] = opcode[1:]
                elif opcode in ['.F32', '.F64', '.F16']:
                    modifiers['float_type'] = opcode[1:]
                elif opcode in ['.RN', '.RZ', '.RM', '.RP']:
                    modifiers['rounding'] = opcode[1:]
                elif opcode in ['.RCP', '.RSQ', '.SIN', '.COS', '.EX2', '.LG2']:
                    modifiers['function'] = opcode[1:]
                elif opcode.startswith('.LUT'):
                    # Extract LUT value from modifier like .LUT0x3c
                    modifiers['lut'] = int(opcode[4:], 16)
        return modifiers
    
    def ExtractLUTFromOperands(self):
        """Extract LUT value from LOP3 operands"""
        # LOP3 typically has format: LOP3.LUT Rd, Rs1, Rs2, Rs3, LUT_value
        # The LUT value is often the 5th operand
        if len(self._operands) >= 5:
            lut_operand = self._operands[4]
            if hasattr(lut_operand, 'Value'):
                return lut_operand.Value
        # Also check opcodes for LUT value
        for opcode in self._opcodes:
            if opcode.startswith('0x'):
                return int(opcode, 16)
        return 0x3c  # Default fallback


    def dump(self):
        print("inst: ", self._id, self._opcodes)
        for operand in self._operands:
            operand.dump()
