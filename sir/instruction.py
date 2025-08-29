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
    def __init__(self, id, opcodes, operands, inst_content, parentBB, pflag=None):
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

        self.Users = set()
        self.ReachingDefs = {}


        # Def/use operands layout correction
        self._UseOpStartIdx = 1

        # IMAD.WIDE has two defs, RN+1:RN
        if "WIDE" in self.opcodes:
            RegPair = self._operands[0].Clone()
            RegPair.SetReg('R' + str(int(RegPair.Reg[1:]) + 1))
            self._operands.insert(1, RegPair)
            self._UseOpStartIdx = 2

        # Store and Branch have no def op
        if self.IsBranch() or self.IsStore():
            self._UseOpStartIdx = 0
        # instruction with predicate carry out have two def op
        elif len(self._operands) > 1 and self._operands[0].IsReg and self._operands[1].IsPredicateReg:
            self._UseOpStartIdx = 2

        
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
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["LDG", "LD"])

    def IsStore(self):
        return len(self._opcodes) > 0 and (self._opcodes[0] in ["STG", "SUST", "ST"])

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
                    Regs[Operand.GetIRRegName(lifter)] = Operand

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

    # Get use operand
    def GetUses(self):
        return self._operands[self._UseOpStartIdx:]
    
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
        return self.__str__()

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

        def _get_val(op, name=""):
            if op.IsRZ:
                return lifter.ir.Constant(op.GetIRType(lifter), 0)
            if op.IsPT:
                return lifter.ir.Constant(op.GetIRType(lifter), 1)
            if op.IsReg:
                val = IRRegs[op.GetIRRegName(lifter)]
                if op.IsNegativeReg:
                    val = IRBuilder.neg(val, f"{name}_neg")
                if op.IsNotReg:
                    val = IRBuilder.not_(val, f"{name}_not")
                if op.IsAbsReg:
                    raise UnsupportedInstructionException(f"Absolute registers not yet supported")
                return val
            if op.IsArg:
                    return ConstMem[op.ArgOffset]
            if op.IsImmediate:
                return lifter.ir.Constant(op.GetIRType(lifter), op.ImmediateValue)
            raise UnsupportedInstructionException(f"Unsupported operand type: {op}")

        if opcode == "MOV" or opcode == "MOV64":
            dest, src = self._operands[0], self._operands[1]
            val = _get_val(src, "mov")
            IRRegs[dest.GetIRRegName(lifter)] = val

        elif opcode == "MOV32I":
            dest, src = self._operands[0], self._operands[1]
            if not src.IsImmediate:
                raise UnsupportedInstructionException(f"MOV32I expects immediate, got: {src}")
            val = lifter.ir.Constant(src.GetIRType(lifter), src.ImmediateValue)
            IRRegs[dest.GetIRRegName(lifter)] = val

        elif opcode == "SETZERO":
            dest = self._operands[0]
            zero_val = lifter.ir.Constant(dest.GetIRType(lifter), 0)
            IRRegs[dest.GetIRRegName(lifter)] = zero_val

        elif opcode == "IMAD" or opcode == "IMAD64":
            dest, op1, op2, op3 = self._operands[0], self._operands[1], self._operands[2], self._operands[3]
            v1 = _get_val(op1, "imad_lhs")
            v2 = _get_val(op2, "imad_rhs")
            v3 = _get_val(op3, "imad_addend")


            tmp = IRBuilder.mul(v1, v2, "imad_tmp")
            tmp = IRBuilder.add(tmp, v3, "imad")
            IRRegs[dest.GetIRRegName(lifter)] = tmp
            
        elif opcode == "EXIT":
            IRBuilder.ret_void()

        elif opcode == "FADD":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "fadd_lhs")
            v2 = _get_val(op2, "fadd_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.fadd(v1, v2, "fadd")

        elif opcode == "FFMA":
            dest, op1, op2, op3 = self._operands[0], self._operands[1], self._operands[2], self._operands[3]
            v1 = _get_val(op1, "ffma_lhs")
            v2 = _get_val(op2, "ffma_rhs")
            v3 = _get_val(op3, "ffma_addend")
            tmp = IRBuilder.fmul(v1, v2, "ffma_tmp")
            tmp = IRBuilder.fadd(tmp, v3, "ffma")
            IRRegs[dest.GetIRRegName(lifter)] = tmp

        elif opcode == "ISCADD":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "iscadd_lhs")
            v2 = _get_val(op2, "iscadd_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(v1, v2, "iscadd")

        elif opcode == "IADD3":
            dest, op1, op2, op3 = (
                self._operands[0], self._operands[1],
                self._operands[2], self._operands[3]
            )
            v1 = _get_val(op1, "iadd3_o1")
            v2 = _get_val(op2, "iadd3_o2")
            v3 = _get_val(op3, "iadd3_o3")
            tmp = IRBuilder.add(v1, v2, "iadd3_tmp")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(tmp, v3, "iadd3")

        elif opcode == "ISUB":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "isub_lhs")
            v2 = _get_val(op2, "isub_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(v1, v2, "sub")
                    
        elif opcode == "SHL":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "shl_lhs")
            v2 = _get_val(op2, "shl_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.shl(v1, v2, "shl")

        elif opcode == "SHL64":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "shl_lhs_64")
            v2 = _get_val(op2, "shl_rhs_64")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.shl(v1, v2, "shl")

        elif opcode == "SHR":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "shr_lhs")
            v2 = _get_val(op2, "shr_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.lshr(v1, v2, "shr")

        elif opcode == "SHF":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "shf_lhs")
            v2 = _get_val(op2, "shf_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.shl(v1, v2, "shf")
                
        elif opcode == "IADD":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "iadd_lhs")
            v2 = _get_val(op2, "iadd_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(v1, v2, "iadd")
        
        elif opcode == "IADD64":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "iadd_lhs")
            v2 = _get_val(op2, "iadd_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(v1, v2, "iadd")

        elif opcode == "IADD32I" or opcode == "IADD32I64":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "iadd32i_lhs")
            
            # TODO: temporary fix
            def sx(v, n):
                v &= (1 << n) - 1
                return (v ^ (1 << (n-1))) - (1 << (n-1))
            op2._ImmediateValue = sx(int(op2.Name, 16), 24)
            v2 = _get_val(op2, "iadd32i_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(v1, v2, "iadd32i")

        elif opcode == "PHI" or opcode == "PHI64":
            dest = self._operands[0]
            phi_val = IRBuilder.phi(dest.GetIRType(lifter), "phi")

            # Some values may be unknown at this point
            # Don't add incoming values yet

            IRRegs[dest.GetIRRegName(lifter)] = phi_val

        elif opcode == "S2R":
            dest, valop = self._operands[0], self._operands[1]
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
            IRRegs[dest.GetIRRegName(lifter)] = val
                
        elif opcode == "LDG":
            dest, ptr = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "ldg_addr")
            val = IRBuilder.load(addr, "ldg")
            IRRegs[dest.GetIRRegName(lifter)] = val
                
        elif opcode == "STG":
            ptr, val = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "stg_addr")
            v = _get_val(val, "stg_val")
            IRBuilder.store(v, addr)

        elif opcode == "FMUL":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "fmul_lhs")
            v2 = _get_val(op2, "fmul_rhs")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.mul(v1, v2, "fmul")

        elif opcode == "INTTOPTR": # psudo instruction placed by # transform/inttoptr.py
            ptr, val = self._operands[0], self._operands[1]
            v1 = _get_val(val, "inttoptr_val")

            IRRegs[ptr.GetIRRegName(lifter)] = IRBuilder.inttoptr(v1, ptr.GetIRType(lifter), "inttoptr")


        elif opcode == "FFMA":
            raise UnsupportedInstructionException

        elif opcode == "LD":
            dest, ptr = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "ld_addr")
            val = IRBuilder.load(addr, "ld")
            IRRegs[dest.GetIRRegName(lifter)] = val
                
        elif opcode == "ST":
            ptr, val = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "st_addr")
            v = _get_val(val, "st_val")
            IRBuilder.store(v, addr)
        
        elif opcode == "LOP":
            dest, a, b = self._operands[0], self._operands[1], self._operands[2]
            subop = self._opcodes[1] if len(self._opcodes) > 1 else None
            vb = _get_val(b, "lop_b")

            if subop == "PASS_B":
                IRRegs[dest.GetIRRegName(lifter)] = vb
            else:
                raise UnsupportedInstructionException
        
        elif opcode == "LOP32I":
            dest, a, b = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(a, "lop32i_a")
            v2 = _get_val(b, "lop32i_b")
            func = self._opcodes[1] if len(self._opcodes) > 1 else None

            if func == "AND":
                IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.and_(v1, v2, "lop32i_and")
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
            targetId = self.GetUses()[0].Name.zfill(4)
            for BB, IRBB in BlockMap.items():
                if BB.addr_content != targetId:
                    continue
                targetBB = IRBB
            
            IRBuilder.branch(targetBB)
        
        elif opcode == "BRK":
            raise UnsupportedInstructionException
        
        elif opcode == "IMNMX":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "imnmx_lhs")
            v2 = _get_val(op2, "imnmx_rhs")

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

            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.select(cond, v1, v2, "imnmx_max")
                
        elif opcode == "PSETP":
            raise UnsupportedInstructionException
        
        elif opcode == "PBK":
            raise UnsupportedInstructionException
             
        elif opcode == "LEA" or opcode == "LEA64":
            dest, op1, op2, op3 = self._operands[0], self._operands[1], self._operands[2], self._operands[3]

            v1 = _get_val(op1, "lea_a")
            v2 = _get_val(op2, "lea_b")
            v3 = _get_val(op3, "lea_scale")

            tmp = IRBuilder.shl(v1, v3, "lea_tmp")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.add(tmp, v2, "lea")
                    
        elif opcode == "F2I":
            dest, op1 = self._operands[0], self._operands[1]

            isUnsigned = "U32" in self._opcodes

            if isUnsigned:
                val = IRBuilder.fptoui(IRRegs[op1.GetIRRegName(lifter)], dest.GetIRType(lifter), "f2i")
            else:
                val = IRBuilder.fptosi(IRRegs[op1.GetIRRegName(lifter)], dest.GetIRType(lifter), "f2i")
            IRRegs[dest.GetIRRegName(lifter)] = val
                    
        elif opcode == "I2F":
            dest, op1 = self._operands[0], self._operands[1]

            isUnsigned = "U32" in self._opcodes

            if isUnsigned:
                val = IRBuilder.uitofp(IRRegs[op1.GetIRRegName(lifter)], dest.GetIRType(lifter), "i2f")
            else:
                val = IRBuilder.sitofp(IRRegs[op1.GetIRRegName(lifter)], dest.GetIRType(lifter), "i2f")
            IRRegs[dest.GetIRRegName(lifter)] = val
                    
        elif opcode == "MUFU":
            dest, src = self._operands[0], self._operands[1]
            func = self._opcodes[1] if len(self._opcodes) > 1 else None
            v = _get_val(src, "mufu_src")

            if func == "RCP": # 1/v
                one = lifter.ir.Constant(dest.GetIRType(lifter), 1.0)
                res = IRBuilder.fdiv(one, v, "mufu_rcp")
                IRRegs[dest.GetIRRegName(lifter)] = res
            else:
                raise UnsupportedInstructionException
                    
        elif opcode == "IABS":
            raise UnsupportedInstructionException

                    
        elif opcode == "LOP3":
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
                    IRRegs[dest.GetIRRegName(lifter)] = pred
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
                    IRRegs[dest.GetIRRegName(lifter)] = pred
            else:
                IRRegs[dest.GetIRRegName(lifter)] = res32


        elif opcode == "MOVM":
            raise UnsupportedInstructionException
                    
        elif opcode == "HMMA":
            raise UnsupportedInstructionException
        
        elif opcode == "DEPBAR":
            pass
        
        elif opcode == "CS2R":
            # CS2R (Convert Special Register to Register)
            ResOp = self._operands[0]
            ValOp = self._operands[1]
            if ResOp.IsReg:
                IRResOp = IRRegs[ResOp.GetIRRegName(lifter)]
                
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

        elif opcode == "ISETP":
            
            r = _get_val(self._operands[2], "branch_operand_0")

            # Remove U32 from opcodes
            # Currently just assuming every int are signed. May be dangerous?  
            opcodes = [opcode for opcode in self._opcodes if opcode != "U32"]
            isUnsigned = "U32" in self._opcodes

            for i in range(1, len(opcodes)):
                temp = _get_val(self._operands[i + 2], f"branch_operand_{i}")
                if opcodes[i] == "AND":
                    r = IRBuilder.and_(r, temp, f"branch_and_{i}")
                else:
                    if isUnsigned:
                        r = IRBuilder.icmp_unsigned(lifter.GetCmpOp(opcodes[i]), r, temp, f"branch_cmp_{i}")
                    else:
                        r = IRBuilder.icmp_signed(lifter.GetCmpOp(opcodes[i]), r, temp, f"branch_cmp_{i}")

            pred = self._operands[0]
            IRRegs[pred.GetIRRegName(lifter)] = r

        elif opcode == "PBRA":
            pred = self._operands[0]

            cond = _get_val(pred, "cond")

            TrueBr, FalseBr = self.parent.GetBranchPair(self)
            IRBuilder.cbranch(cond, BlockMap[TrueBr], BlockMap[FalseBr])

        elif opcode == "PACK64":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1)
            v2 = _get_val(op2)
            lo64 = IRBuilder.zext(v1, ir.IntType(64), "pack64_lo")
            hi64 = IRBuilder.zext(v2, ir.IntType(64), "pack64_hi")
            hiShift = IRBuilder.shl(hi64, ir.Constant(ir.IntType(64), 32), "pack64_hi_shift")
            packed = IRBuilder.or_(lo64, hiShift, "pack64_result")
            IRRegs[dest.GetIRRegName(lifter)] = packed

        elif opcode == "CAST64":
            dest, op1 = self._operands[0], self._operands[1]
            if not dest.IsReg:
                raise UnsupportedInstructionException(f"CAST64 expects a register operand, got: {dest}")

            val = _get_val(op1)

            val64 = IRBuilder.sext(val, ir.IntType(64), "cast64")
            IRRegs[dest.GetIRRegName(lifter)] = val64

        elif opcode == "BITCAST":
            dest, src = self._operands[0], self._operands[1]
            dest_type = self._operands[0].GetIRType(lifter)

            val = _get_val(src)

            cast_val = IRBuilder.bitcast(val, dest_type, "cast")
            IRRegs[dest.GetIRRegName(lifter)] = cast_val

        else:
            print("lift instruction: ", opcode)
            raise UnsupportedInstructionException 

            
    def dump(self):
        print("inst: ", self._id, self._opcodes)
        for operand in self._operands:
            operand.dump()
