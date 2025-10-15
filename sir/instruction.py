from sir.operand import Operand
from sir.controlcode import ControlCode
from sir.controlcode import PresetCtlCodeException
from llvmlite import ir

class UnsupportedOperatorException(Exception):
    pass

class UnsupportedInstructionException(Exception):
    pass

class InvalidTypeException(Exception):
    pass

class Instruction:
    def __init__(self, id, opcodes, operands, parentBB=None, inst_content=None, pflag=None):
        self._id = id
        self._opcodes = opcodes
        self._operands = operands
        self._InstContent = inst_content
        self._TwinIdx = ""
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
            inst_content=self._InstContent,
            pflag=cloned_pflag
        )
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
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["LDG", "LD", "LDS", "ULDC"])
    
    def IsGlobalLoad(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["LDG"])

    def IsStore(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["STG", "SUST", "ST", "STS"])
    
    def IsGlobalStore(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["STG"])

    # Set all operands as skipped
    def SetSkip(self):
        for Operand in self._operands:
            Operand.SetSkip()
           
    # Collect registers used in instructions
    def GetRegs(self, Regs, lifter):
        # Check twin instruction case
        TwinIdx = self._TwinIdx
        
        for Operand in self._operands:
            if Operand.IsReg:
                if TwinIdx.find(Operand.Reg) == 0:
                    if not Operand.TypeDesc == "NOTYPE":
                        RegName = TwinIdx + Operand.TypeDesc
                        Regs[RegName] = Operand
                else:
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


    def Lift(self, lifter, IRBuilder: ir.IRBuilder, IRRegs, ConstMem, BlockMap):
        if len(self._opcodes) == 0:
            raise UnsupportedInstructionException("Empty opcode list")
        opcode = self._opcodes[0]

        def roughSearch(op):
            reg = op.Reg
            name = op.GetIRName(lifter)
            targetType = name.replace(reg, "")

            bestKey = max(IRRegs.keys(), key=lambda k: (k.startswith(reg), len(k)))

            val = IRBuilder.bitcast(IRRegs[bestKey], op.GetIRType(lifter), f"{name}_cast")

            return val

        def _get_val(op, name=""):
            if op.IsRZ:
                return lifter.ir.Constant(op.GetIRType(lifter), 0)
            if op.IsPT:
                return lifter.ir.Constant(op.GetIRType(lifter), 1)
            if op.IsReg:
                irName = op.GetIRName(lifter)
                if irName not in IRRegs:
                    val = roughSearch(op)
                else:
                    val = IRRegs[irName]
                    
                if op.IsNegativeReg:
                    if op.GetTypeDesc().startswith('F'):
                        val = IRBuilder.fneg(val, f"{name}_fneg")
                    else:
                        val = IRBuilder.neg(val, f"{name}_neg")
                if op.IsNotReg:
                    val = IRBuilder.not_(val, f"{name}_not")
                if op.IsAbsReg:
                    val = IRBuilder.call(lifter.DeviceFuncs["abs"], [val], f"{name}_abs")
                return val
            if op.IsArg:
                    return ConstMem[op.GetIRName(lifter)]
            if op.IsImmediate:
                return lifter.ir.Constant(op.GetIRType(lifter), op.ImmediateValue)
            raise UnsupportedInstructionException(f"Unsupported operand type: {op}")

        if opcode == "MOV" or opcode == "MOV64" or opcode == "UMOV":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            val = _get_val(src, "mov")
            IRRegs[dest.GetIRName(lifter)] = val

        elif opcode == "MOV32I":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            if not src.IsImmediate:
                raise UnsupportedInstructionException(f"MOV32I expects immediate, got: {src}")
            val = lifter.ir.Constant(src.GetIRType(lifter), src.ImmediateValue)
            IRRegs[dest.GetIRName(lifter)] = val

        elif opcode == "SETZERO":
            dest = self.GetDefs()[0]
            zero_val = lifter.ir.Constant(dest.GetIRType(lifter), 0)
            IRRegs[dest.GetIRName(lifter)] = zero_val
            
        elif opcode == "IMAD":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "imad_lhs")
            v2 = _get_val(uses[1], "imad_rhs")
            v3 = _get_val(uses[2], "imad_addend")


            tmp = IRBuilder.mul(v1, v2, "imad_tmp")
            tmp = IRBuilder.add(tmp, v3, "imad")
            IRRegs[dest.GetIRName(lifter)] = tmp

        elif opcode == "IMAD64":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "imad_lhs")
            v1_64 = IRBuilder.sext(v1, lifter.ir.IntType(64), "imad_lhs_64")
            v2 = _get_val(uses[1], "imad_rhs")
            v2_64 = IRBuilder.sext(v2, lifter.ir.IntType(64), "imad_rhs_64")
            v3 = _get_val(uses[2], "imad_addend")


            tmp = IRBuilder.mul(v1_64, v2_64, "imad_tmp")
            tmp = IRBuilder.add(tmp, v3, "imad")
            IRRegs[dest.GetIRName(lifter)] = tmp
            
        elif opcode == "EXIT":
            IRBuilder.ret_void()

        elif opcode == "FADD":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "fadd_lhs")
            v2 = _get_val(uses[1], "fadd_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.fadd(v1, v2, "fadd")

        elif opcode == "FFMA":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "ffma_lhs")
            v2 = _get_val(uses[1], "ffma_rhs")
            v3 = _get_val(uses[2], "ffma_addend")
            tmp = IRBuilder.fmul(v1, v2, "ffma_tmp")
            tmp = IRBuilder.fadd(tmp, v3, "ffma")
            IRRegs[dest.GetIRName(lifter)] = tmp

        elif opcode == "ISCADD":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "iscadd_lhs")
            v2 = _get_val(uses[1], "iscadd_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(v1, v2, "iscadd")

        elif opcode == "IADD3" or opcode == "UIADD3" or opcode == "IADD364":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "iadd3_o1")
            v2 = _get_val(uses[1], "iadd3_o2")
            v3 = _get_val(uses[2], "iadd3_o3")
            tmp = IRBuilder.add(v1, v2, "iadd3_tmp")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(tmp, v3, "iadd3")

        elif opcode == "ISUB":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "isub_lhs")
            v2 = _get_val(uses[1], "isub_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(v1, v2, "sub")
                    
        elif opcode == "SHL":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "shl_lhs")
            v2 = _get_val(uses[1], "shl_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.shl(v1, v2, "shl")

        elif opcode == "SHL64":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "shl_lhs_64")
            v2 = _get_val(uses[1], "shl_rhs_64")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.shl(v1, v2, "shl")

        elif opcode == "SHR" or opcode == "SHR64":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "shr_lhs")
            v2 = _get_val(uses[1], "shr_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.lshr(v1, v2, "shr")

        elif opcode == "SHF" or opcode == "USHF":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "shf_lhs")
            v2 = _get_val(uses[1], "shf_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.shl(v1, v2, "shf")
                
        elif opcode == "IADD":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "iadd_lhs")
            v2 = _get_val(uses[1], "iadd_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(v1, v2, "iadd")
            
        elif opcode == "SEL" or opcode == "FSEL":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "sel_true")
            v2 = _get_val(uses[1], "sel_false")
            pred = _get_val(uses[2], "sel_pred")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.select(pred, v1, v2, "sel")
        
        elif opcode == "IADD64":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "iadd_lhs")
            v2 = _get_val(uses[1], "iadd_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(v1, v2, "iadd")

        elif opcode == "IADD32I" or opcode == "IADD32I64":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            op1, op2 = uses[0], uses[1]
            v1 = _get_val(op1, "iadd32i_lhs")
            
            # TODO: temporary fix
            def sx(v, n):
                v &= (1 << n) - 1
                return (v ^ (1 << (n-1))) - (1 << (n-1))
            op2._ImmediateValue = sx(int(op2.Name, 16), 24)
            v2 = _get_val(op2, "iadd32i_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(v1, v2, "iadd32i")

        elif opcode == "PHI" or opcode == "PHI64":
            dest = self.GetDefs()[0]
            phi_val = IRBuilder.phi(dest.GetIRType(lifter), "phi")

            # Some values may be unknown at this point
            # Don't add incoming values yet

            IRRegs[dest.GetIRName(lifter)] = phi_val

        elif opcode == "S2R":
            dest = self.GetDefs()[0]
            valop = self.GetUses()[0]
            if valop.IsThreadIdx:
                val = IRBuilder.call(lifter.GetThreadIdx, [], "ThreadIdx")
            elif valop.IsBlockDim:
                val = IRBuilder.call(lifter.GetBlockDim, [], "BlockDim")
            elif valop.IsBlockIdx:
                val = IRBuilder.call(lifter.GetBlockIdx, [], "BlockIdx")
            elif valop.IsLaneId:
                val = IRBuilder.call(lifter.GetLaneId, [], "LaneId")
            elif valop.IsWarpId:
                val = IRBuilder.call(lifter.GetWarpId, [], "WarpId")
            else:
                print(f"S2R: Unknown special register {valop.Name}")
                val = lifter.ir.Constant(lifter.ir.IntType(32), 0)
            IRRegs[dest.GetIRName(lifter)] = val
                
        elif opcode == "LDG" or opcode == "LDG64":
            dest = self.GetDefs()[0]
            ptr = self.GetUses()[0]
            addr = _get_val(ptr, "ldg_addr")
            val = IRBuilder.load(addr, "ldg", typ=dest.GetIRType(lifter))
            IRRegs[dest.GetIRName(lifter)] = val
                
        elif opcode == "STG":
            uses = self.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "stg_addr")
            v = _get_val(val, "stg_val")
            IRBuilder.store(v, addr)

        elif opcode == "LDS":
            dest = self.GetDefs()[0]
            ptr = self.GetUses()[0]
            addr = _get_val(ptr, "lds_addr")
            addr = IRBuilder.gep(lifter.SharedMem, [lifter.ir.Constant(lifter.ir.IntType(32), 0), addr], "lds_shared_addr")
            val = IRBuilder.load(addr, "lds")
            IRRegs[dest.GetIRName(lifter)] = val

        elif opcode == "STS":
            uses = self.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "sts_addr")
            addr = IRBuilder.gep(lifter.SharedMem, [lifter.ir.Constant(lifter.ir.IntType(32), 0), addr], "sts_shared_addr")
            v = _get_val(val, "sts_val")
            IRBuilder.store(v, addr)

        elif opcode == "FMUL":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "fmul_lhs")
            v2 = _get_val(uses[1], "fmul_rhs")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.fmul(v1, v2, "fmul")

        elif opcode == "INTTOPTR": # psudo instruction placed by # transform/inttoptr.py
            ptr = self.GetDefs()[0]
            val = self.GetUses()[0]
            v1 = _get_val(val, "inttoptr_val")

            IRRegs[ptr.GetIRName(lifter)] = IRBuilder.inttoptr(v1, ptr.GetIRType(lifter), "inttoptr")


        elif opcode == "FFMA":
            raise UnsupportedInstructionException

        elif opcode == "LD":
            dest = self.GetDefs()[0]
            ptr = self.GetUses()[0]
            addr = _get_val(ptr, "ld_addr")
            val = IRBuilder.load(addr, "ld")
            IRRegs[dest.GetIRName(lifter)] = val
                
        elif opcode == "ST":
            uses = self.GetUses()
            ptr, val = uses[0], uses[1]
            addr = _get_val(ptr, "st_addr")
            v = _get_val(val, "st_val")
            IRBuilder.store(v, addr)
        
        elif opcode == "LOP":
            dest = self.GetDefs()[0]
            a, b = self.GetUses()[0], self.GetUses()[1]
            subop = self._opcodes[1] if len(self._opcodes) > 1 else None
            vb = _get_val(b, "lop_b")

            if subop == "PASS_B":
                IRRegs[dest.GetIRName(lifter)] = vb
            else:
                raise UnsupportedInstructionException
        
        elif opcode == "LOP32I":
            dest = self.GetDefs()[0]
            a, b = self.GetUses()[0], self.GetUses()[1]
            v1 = _get_val(a, "lop32i_a")
            v2 = _get_val(b, "lop32i_b")
            func = self._opcodes[1] if len(self._opcodes) > 1 else None

            if func == "AND":
                IRRegs[dest.GetIRName(lifter)] = IRBuilder.and_(v1, v2, "lop32i_and")
            else:
                raise UnsupportedInstructionException
        
        elif opcode == "BFE":
            raise UnsupportedInstructionException
        
        elif opcode == "BFI":
            raise UnsupportedInstructionException
        
        elif opcode == "SSY":
            pass

        elif opcode == "SYNC":
            pass
        
        elif opcode == "BRA":
            targetId = self.GetUses()[-1].Name.zfill(4)
            for BB, IRBB in BlockMap.items():
                if int(BB.addr_content ,16) != int(targetId, 16):
                    continue
                targetBB = IRBB
            
            IRBuilder.branch(targetBB)
        
        elif opcode == "BRK":
            raise UnsupportedInstructionException
        
        elif opcode == "IMNMX":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0], "imnmx_lhs")
            v2 = _get_val(uses[1], "imnmx_rhs")

            isUnsigned = "U32" in self._opcodes
            isMax = "MXA" in self._opcodes

            if isUnsigned:
                if isMax:
                    cond = IRBuilder.icmp_unsigned('>', v1, v2, "imnmx_cmp")
                else:
                    cond = IRBuilder.icmp_unsigned('<', v1, v2, "imnmx_cmp")
            else:
                if isMax:
                    cond = IRBuilder.icmp_signed('>', v1, v2, "imnmx_cmp")
                else:
                    cond = IRBuilder.icmp_signed('<', v1, v2, "imnmx_cmp")

            IRRegs[dest.GetIRName(lifter)] = IRBuilder.select(cond, v1, v2, "imnmx_max")
                
        elif opcode == "PSETP":
            raise UnsupportedInstructionException
        
        elif opcode == "PBK":
            raise UnsupportedInstructionException
             
        elif opcode == "LEA" or opcode == "LEA64" or opcode == "ULEA":
            dest = self.GetDefs()[0]
            uses = self.GetUses()

            v1 = _get_val(uses[0], "lea_a")
            v2 = _get_val(uses[1], "lea_b")
            v3 = _get_val(uses[2], "lea_scale")

            tmp = IRBuilder.shl(v1, v3, "lea_tmp")
            IRRegs[dest.GetIRName(lifter)] = IRBuilder.add(tmp, v2, "lea")
                    
        elif opcode == "F2I":
            dest = self.GetDefs()[0]
            op1 = self.GetUses()[0]

            isUnsigned = "U32" in self._opcodes

            v1 = _get_val(op1, "f2i_src")

            if isUnsigned:
                val = IRBuilder.fptoui(v1, dest.GetIRType(lifter), "f2i")
            else:
                val = IRBuilder.fptosi(v1, dest.GetIRType(lifter), "f2i")
            IRRegs[dest.GetIRName(lifter)] = val
                    
        elif opcode == "I2F":
            dest = self.GetDefs()[0]
            op1 = self.GetUses()[0]

            isUnsigned = "U32" in self._opcodes

            v1 = _get_val(op1, "i2f_src")

            if isUnsigned:
                val = IRBuilder.uitofp(v1, dest.GetIRType(lifter), "i2f")
            else:
                val = IRBuilder.sitofp(v1, dest.GetIRType(lifter), "i2f")
            IRRegs[dest.GetIRName(lifter)] = val
                    
        elif opcode == "MUFU":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            func = self._opcodes[1] if len(self._opcodes) > 1 else None
            v = _get_val(src, "mufu_src")

            if func == "RCP": # 1/v
                one = lifter.ir.Constant(dest.GetIRType(lifter), 1.0)
                res = IRBuilder.fdiv(one, v, "mufu_rcp")
                IRRegs[dest.GetIRName(lifter)] = res
            else:
                raise UnsupportedInstructionException
                    
        elif opcode == "IABS":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            v = _get_val(src, "iabs_src")
            res = IRBuilder.call(lifter.DeviceFuncs["abs"], [v], "iabs")
            IRRegs[dest.GetIRName(lifter)] = res

                    
        elif opcode == "LOP3" or opcode == "ULOP3" or opcode == "PLOP3":
            # Lower LOP3.LUT for both register and predicate destinations.
            dest = self.GetDefs()[0]
            src1, src2, src3 = self.GetUses()[0], self.GetUses()[1], self.GetUses()[2]
            func = self._opcodes[1] if len(self._opcodes) > 1 else None

            if func != "LUT":
                raise UnsupportedInstructionException

            # Find the LUT immediate among uses (last immediate before any PT operand)
            imm8 = None
            for op in reversed(self.GetUses()):
                if op.IsImmediate:
                    imm8 = op.ImmediateValue & 0xFF
                    break
            if imm8 is None:
                raise UnsupportedInstructionException

            a = _get_val(src1, "lop3_a")
            b = _get_val(src2, "lop3_b")
            c = _get_val(src3, "lop3_c")

            # Sum-of-products bitwise construction for 32-bit result
            zero = lifter.ir.Constant(b.type, 0)
            # Fast-path a-agnostic mask used frequently: imm8 == 0xC0 -> b & c
            if imm8 == 0xC0:
                res32 = IRBuilder.and_(b, c, "lop3_bc")
            else:
                nota = IRBuilder.not_(a, "lop3_nota")
                notb = IRBuilder.not_(b, "lop3_notb")
                notc = IRBuilder.not_(c, "lop3_notc")
                res32 = zero
                for idx in range(8):
                    if ((imm8 >> idx) & 1) == 0:
                        continue
                    xa = a if (idx & 1) else nota
                    xb = b if (idx & 2) else notb
                    xc = c if (idx & 4) else notc
                    tmp = IRBuilder.and_(xa, xb, f"lop3_and_ab_{idx}")
                    tmp = IRBuilder.and_(tmp, xc, f"lop3_and_abc_{idx}")
                    res32 = IRBuilder.or_(res32, tmp, f"lop3_or_{idx}")

            if dest.IsPredicateReg:
                # For P-dest, prefer a safe minimal lowering:
                #  - Recognize imm8==0xC0 -> (B & C) != 0 (matches loop3).
                #  - Otherwise, approximate with 1-bit LUT lookup of (LSB(A),LSB(B),LSB(C)).
                uses = self.GetUses()
                c_is_imm = len(uses) >= 3 and uses[2].IsImmediate
                if imm8 == 0xC0 and c_is_imm:
                    mask = IRBuilder.and_(b, c, "lop3_bc_mask")
                    pred = IRBuilder.icmp_unsigned("!=", mask, zero, "lop3_pred")
                    IRRegs[dest.GetIRName(lifter)] = pred
                else:
                    one_i32 = lifter.ir.Constant(a.type, 1)
                    a_lsb_i32 = IRBuilder.and_(a, one_i32, "lop3_a_lsb")
                    b_lsb_i32 = IRBuilder.and_(b, one_i32, "lop3_b_lsb")
                    c_lsb_i32 = IRBuilder.and_((c if c.type == a.type else IRBuilder.zext(c, a.type, "lop3_c_zext")), one_i32, "lop3_c_lsb")
                    a0 = IRBuilder.icmp_unsigned("!=", a_lsb_i32, lifter.ir.Constant(a.type, 0), "lop3_a0")
                    b0 = IRBuilder.icmp_unsigned("!=", b_lsb_i32, lifter.ir.Constant(b.type, 0), "lop3_b0")
                    c0 = IRBuilder.icmp_unsigned("!=", c_lsb_i32, lifter.ir.Constant(a.type, 0), "lop3_c0")
                    a0_i32 = IRBuilder.sext(a0, a.type)
                    b0_i32 = IRBuilder.sext(b0, b.type)
                    c0_i32 = IRBuilder.sext(c0, a.type)
                    idx = IRBuilder.or_(
                        a0_i32,
                        IRBuilder.or_(
                            IRBuilder.shl(b0_i32, lifter.ir.Constant(b.type, 1)),
                            IRBuilder.shl(c0_i32, lifter.ir.Constant(a.type, 2))
                        ),
                        "lop3_idx"
                    )
                    imm_i32 = lifter.ir.Constant(a.type, imm8 & 0xFF)
                    bit_i32 = IRBuilder.and_(
                        IRBuilder.lshr(imm_i32, idx, "lop3_lut_shift"),
                        lifter.ir.Constant(a.type, 1),
                        "lop3_lut_bit"
                    )
                    pred = IRBuilder.icmp_unsigned("!=", bit_i32, lifter.ir.Constant(a.type, 0), "lop3_pred")
                    IRRegs[dest.GetIRName(lifter)] = pred
            else:
                IRRegs[dest.GetIRName(lifter)] = res32


        elif opcode == "MOVM":
            # TODO: dummy implementation
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            val = _get_val(src, "movm")
            IRRegs[dest.GetIRName(lifter)] = val
                    
        elif opcode == "HMMA":
            
            size = self.opcodes[1]
            type = self.opcodes[2]

            if type != "F32":
                raise UnsupportedInstructionException
            
            if size != "1688":
                raise UnsupportedInstructionException

            if "hmma1688f32" not in lifter.DeviceFuncs:
                # hmma1688f32(float*8) -> float*4
                outputType = ir.LiteralStructType([ir.FloatType()] * 4)
                inputType = [ir.FloatType()] * 8
                func_ty = ir.FunctionType(outputType, inputType)
                lifter.DeviceFuncs["hmma1688f32"] = ir.Function(IRBuilder.module, func_ty, name="hmma1688f32")

            func = lifter.DeviceFuncs["hmma1688f32"]
            uses = self.GetUses()
            args = [_get_val(uses[i], f"hmma_arg_{i}") for i in range(8)]
            val = IRBuilder.call(func, args, "hmma_call")
            
            # # unpack into 4 dest registers
            # dests = self.GetDefs()
            # for i in range(4):
            #     IRRegs[dests[i].GetIRName(lifter)] = IRBuilder.extract_value(val, i, f"hmma_res_{i}")
        
        elif opcode == "DEPBAR":
            pass

        elif opcode == "ULDC" or opcode == "ULDC64":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            val = _get_val(src, "uldc")
            IRRegs[dest.GetIRName(lifter)] = val
        
        elif opcode == "CS2R":
            # CS2R (Convert Special Register to Register)
            ResOp = self.GetDefs()[0]
            ValOp = self.GetUses()[0]
            if ResOp.IsReg:
                IRResOp = IRRegs[ResOp.GetIRName(lifter)]
                
                # Determine which special register this is using the new approach
                if ValOp.IsThreadIdx:
                    IRVal = IRBuilder.call(lifter.GetThreadIdx, [], "cs2r_tid")
                elif ValOp.IsBlockDim:
                    IRVal = IRBuilder.call(lifter.GetBlockDim, [], "cs2r_ntid")
                elif ValOp.IsBlockIdx:
                    IRVal = IRBuilder.call(lifter.GetBlockIdx, [], "cs2r_ctaid")
                elif ValOp.IsLaneId:
                    IRVal = IRBuilder.call(lifter.GetLaneId, [], "cs2r_lane")
                elif ValOp.IsWarpId:
                    IRVal = IRBuilder.call(lifter.GetWarpId, [], "cs2r_warp")
                elif ValOp.IsRZ:
                    IRVal = ir.Constant(ir.IntType(32), 0)
                else:
                    print(f"CS2R: Unknown special register {ValOp}")
                    IRVal = ir.Constant(ir.IntType(32), 0)
                
                IRBuilder.store(IRVal, IRResOp)

        elif opcode == "ISETP" or opcode == "ISETP64" or opcode == "FSETP":
            uses = self.GetUses()
            # Some encodings include a leading PT predicate use; skip it
            start_idx = 1 if len(uses) > 0 and getattr(uses[0], 'IsPT', False) else 0
            r = _get_val(uses[start_idx], "branch_operand_0")

            # Remove U32 from opcodes
            # Currently just assuming every int are signed. May be dangerous?  
            opcodes = [opcode for opcode in self._opcodes if opcode != "U32"]
            isUnsigned = "U32" in self._opcodes

            for i in range(1, len(opcodes)):
                temp = _get_val(uses[start_idx + i], f"branch_operand_{i}")
                if opcodes[i] == "AND":
                    r = IRBuilder.and_(r, temp, f"branch_and_{i}")
                elif opcodes[i] == "OR":
                    r = IRBuilder.or_(r, temp, f"branch_or_{i}")
                elif opcodes[i] == "XOR":
                    r = IRBuilder.xor(r, temp, f"branch_xor_{i}")
                elif opcodes[i] == "EX":
                    pass #TODO:?
                else:
                    if isUnsigned:
                        r = IRBuilder.icmp_unsigned(lifter.GetCmpOp(opcodes[i]), r, temp, f"branch_cmp_{i}")
                    else:
                        r = IRBuilder.icmp_signed(lifter.GetCmpOp(opcodes[i]), r, temp, f"branch_cmp_{i}")

            pred = self.GetDefs()[0]
            IRRegs[pred.GetIRName(lifter)] = r

        elif opcode == "PBRA":
            pred = self.GetUses()[0]

            cond = _get_val(pred, "cond")

            TrueBr, FalseBr = self.parent.GetBranchPair(self, BlockMap.keys())
            IRBuilder.cbranch(cond, BlockMap[TrueBr], BlockMap[FalseBr])

        elif opcode == "BAR":
            if len(self._opcodes) > 1 and self._opcodes[1] != "SYNC":
                raise UnsupportedInstructionException

            IRBuilder.call(lifter.DeviceFuncs["syncthreads"], [], "barrier")

        elif opcode == "PACK64":
            dest = self.GetDefs()[0]
            uses = self.GetUses()
            v1 = _get_val(uses[0])
            v2 = _get_val(uses[1])
            lo64 = IRBuilder.zext(v1, ir.IntType(64), "pack64_lo")
            hi64 = IRBuilder.zext(v2, ir.IntType(64), "pack64_hi")
            hiShift = IRBuilder.shl(hi64, ir.Constant(ir.IntType(64), 32), "pack64_hi_shift")
            packed = IRBuilder.or_(lo64, hiShift, "pack64_result")
            IRRegs[dest.GetIRName(lifter)] = packed
            
        elif opcode == "UNPACK64":
            dests = self.GetDefs()
            src = self.GetUses()[0]
            val = _get_val(src)
            lo32 = IRBuilder.trunc(val, ir.IntType(32), "unpack64_lo")
            hi32_shift = IRBuilder.lshr(val, ir.Constant(ir.IntType(64), 32), "unpack64_hi_shift")
            hi32 = IRBuilder.trunc(hi32_shift, ir.IntType(32), "unpack64_hi")
            IRRegs[dests[0].GetIRName(lifter)] = lo32
            IRRegs[dests[1].GetIRName(lifter)] = hi32

        elif opcode == "CAST64":
            dest = self.GetDefs()[0]
            op1 = self.GetUses()[0]
            if not dest.IsReg:
                raise UnsupportedInstructionException(f"CAST64 expects a register operand, got: {dest}")

            val = _get_val(op1)

            val64 = IRBuilder.sext(val, ir.IntType(64), "cast64")
            IRRegs[dest.GetIRName(lifter)] = val64

        elif opcode == "BITCAST":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            dest_type = dest.GetIRType(lifter)

            val = _get_val(src)

            cast_val = IRBuilder.bitcast(val, dest_type, "cast")
            IRRegs[dest.GetIRName(lifter)] = cast_val
            
        elif opcode == "VOTE" or opcode == "VOTEU":
            dest = self.GetDefs()[0]
            pred = _get_val(self.GetUses()[0])
            mask = _get_val(self.GetUses()[1])
            
            mode = self.opcodes[1]
            
            if mode == "ANY":
                funcName = "vote_any"
                if funcName not in lifter.DeviceFuncs:
                    FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(1), ir.IntType(1)], False)
                    lifter.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)
                voteVal = IRBuilder.call(lifter.DeviceFuncs[funcName], [pred, mask], "vote_any")
            else:
                raise UnsupportedInstructionException

            IRRegs[dest.GetIRName(lifter)] = voteVal

        elif opcode == "MATCH":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]

            val = _get_val(src)
            mode = self.opcodes[1]            
            type = self.opcodes[2]

            if mode != 'ANY':
                raise UnsupportedInstructionException

            if type == 'U64':
                dtype = ir.IntType(64)
            elif type == 'U32':
                dtype = ir.IntType(32)
            else:
                raise UnsupportedInstructionException

            funcName = f"match_{mode}_{type}"
            if funcName not in lifter.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [dtype], False)
                lifter.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            matchVal = IRBuilder.call(lifter.DeviceFuncs[funcName], [val], "match")
            IRRegs[dest.GetIRName(lifter)] = matchVal

        elif opcode == "BREV":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            val = _get_val(src)

            if "brev" not in lifter.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32)], False)
                lifter.DeviceFuncs["brev"] = ir.Function(IRBuilder.module, FuncTy, "brev")

            revVal = IRBuilder.call(lifter.DeviceFuncs["brev"], [val], "brev")
            IRRegs[dest.GetIRName(lifter)] = revVal

        elif opcode == "FLO":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            val = _get_val(src)

            type = self.opcodes[1]
            mode = self.opcodes[2]

            if type == 'U64':
                dtype = ir.IntType(64)
                typeName = 'i64'
            elif type == 'U32':
                dtype = ir.IntType(32)
                typeName = 'i32'
            else:
                raise UnsupportedInstructionException

            if mode == 'SH':
                funcName = f"llvm.ctlz.{typeName}"     
            else:
                raise UnsupportedInstructionException

            if funcName not in lifter.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [dtype], False)
                lifter.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            floVal = IRBuilder.call(lifter.DeviceFuncs[funcName], [val], "flo")
            IRRegs[dest.GetIRName(lifter)] = floVal

        elif opcode == "POPC":
            dest = self.GetDefs()[0]
            src = self.GetUses()[0]
            val = _get_val(src)

            funcName = f"llvm.ctpop.i32"     

            if funcName not in lifter.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32)], False)
                lifter.DeviceFuncs[funcName] = ir.Function(IRBuilder.module, FuncTy, funcName)

            popcVal = IRBuilder.call(lifter.DeviceFuncs[funcName], [val], "popc")
            IRRegs[dest.GetIRName(lifter)] = popcVal

        elif opcode == "RED":
            uses = self.GetUses()
            src1, src2 = uses[0], uses[1]
            val1 = _get_val(src1)
            val2 = _get_val(src2)

            mode = self.opcodes[2]
            order = 'seq_cst'
            
            res = IRBuilder.atomic_rmw(mode, val1, val2, order)

        elif opcode == "PRMT":
            dest = self.GetDefs()[0]
            a, sel, b = self.GetUses()[0], self.GetUses()[1], self.GetUses()[2]
            v1 = _get_val(a, "prmt_a")
            v2 = _get_val(b, "prmt_b")
            imm8 = _get_val(sel, "prmt_sel")
            
            # Reinterpret any non-i32 types to i32
            if v1.type != ir.IntType(32):
                v1 = IRBuilder.bitcast(v1, ir.IntType(32), "prmt_a_i32")
            if v2.type != ir.IntType(32):
                v2 = IRBuilder.bitcast(v2, ir.IntType(32), "prmt_b_i32")
            if imm8.type != ir.IntType(32):
                imm8 = IRBuilder.bitcast(imm8, ir.IntType(32), "prmt_sel_i32")

            if "prmt" not in lifter.DeviceFuncs:
                FuncTy = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32), ir.IntType(32)], False)
                lifter.DeviceFuncs["prmt"] = ir.Function(IRBuilder.module, FuncTy, "prmt")

            prmtVal = IRBuilder.call(lifter.DeviceFuncs["prmt"], [v1, v2, imm8], "prmt")
            IRRegs[dest.GetIRName(lifter)] = prmtVal
        
        else:
            print("Unhandled instruction: ", opcode)
            raise UnsupportedInstructionException 

            
    def dump(self):
        print("inst: ", self._id, self._opcodes)
        for operand in self._operands:
            operand.dump()
