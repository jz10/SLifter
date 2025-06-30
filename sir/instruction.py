from sir.operand import Operand
from sir.controlcode import ControlCode
from sir.controlcode import PresetCtlCodeException
from llvmlite import ir

ARG_OFFSET = 320 # 0x140

class UnsupportedOperatorException(Exception):
    pass

class UnsupportedInstructionException(Exception):
    pass

class InvalidTypeException(Exception):
    pass

class Instruction:
    def __init__(self, id, opcodes, operands, inst_content):
        self._id = id
        self._opcodes = opcodes
        self._operands = operands
        self._InstContent = inst_content
        self._TwinIdx = ""
        self._TrueBranch = None
        self._FalseBranch = None
        self._CtlCode = None
        
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
    def pflag(self):
        if self.IsPredicateReg(self._opcodes[0]):
            return self._opcodes[0]
        else:
            return None

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
        if self._opcodes[0] == "EXIT":
            return True
        elif len(self._opcodes) > 1 and self._opcodes[1] == "EXIT":
            return True

        return False

    def IsBranch(self):
        if len(self.operands)>0 and self.GetDef().Reg:
            return self.IsPredicateReg(self.GetDef().Reg)

    def InCondPath(self):
        return self.IsPredicateReg(self._opcodes[0])

    def IsBinary(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "FFMA" or self._opcodes[Idx] == "FADD" or self._opcodes[Idx] == "XMAD" or self._opcodes[Idx] == "IMAD" or self._opcodes[Idx] == "SHL" or self._opcodes[Idx] == "SHR" or self._opcodes[Idx] == "SHF" or self._opcodes[Idx] == "S2R"
        # return self._opcodes[Idx] == "FFMA" or self._opcodes[Idx] == "FADD" or self._opcodes[Idx] == "XMAD" or self._opcodes[Idx] == "IMAD" or self._opcodes[Idx] == "SHL" or self._opcodes[Idx] == "SHR" or self._opcodes[Idx] == "SHF" or self._opcodes[Idx] == "S2R" or self._opcodes[Idx] == "ISCADD"
    
    
    def IsNOP(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "NOP"

    def IsAddrCompute(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        if self._opcodes[Idx] == "IADD":
            # Check operands
            if len(self._operands) == 3:
                operand = self._operands[2]
                # Check function argument operand
                return operand.IsArg

        return False

    def IsLoad(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "LDG" or self._opcodes[Idx] == "SULD"

    def IsStore(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "STG" or self._opcodes[Idx] == "SUST"

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
        
    # Get def operand
    def GetDef(self):
        if len(self._operands) > 0:
            return self._operands[0]
        return None

    def GetRegName(self, Reg):
        return Reg.split('@')[0]
    
    def RenameReg(self, Reg, Inst):
        RegName = self.GetRegName(Reg)
        NewReg = RegName + "@" + str(Inst.id)
        return NewReg


    def ProcessBB(self, BB, InRegsMap, OutRegsMap):

        NewInRegs = self.GenerateInRegs(BB, InRegsMap, OutRegsMap)

        if NewInRegs == InRegsMap[BB]:
            return False
        
        CurrRegs = NewInRegs.copy()

        for Inst in BB.instructions:
            Def = Inst.getDef()
            if self.GetRegName(Def) in CurrRegs:
                CurrRegs[self.GetRegName(Def)] = self.RenameReg(Def, Inst)
                
        
            

        return False


    # Get use operand
    def GetUses(self):
        Uses = []
        if len(self._operands) > 1:
            for i in range(1, len(self._operands)):
                Uses.append(self._operands[i])
        return Uses

    # Get branch flag
    def GetBranchFlag(self):
        Operand = self._operands[0]
        if Operand.Name == "P0":
            return Operand.Name
        else:
            return None
    
    def __str__(self):
        operand_strs = []
        for operand in self._operands:
            if operand.IsMemAddr:
                if operand._MemAddrOffset:
                    operand_strs.append(f"[{operand.Reg}+{operand._MemAddrOffset}]")
                else:
                    operand_strs.append(f"[{operand.Reg}]")
            elif operand.IsReg:
                operand_strs.append(f"{operand.Reg}.{operand._Suffix}" if operand._Suffix else operand.Reg)
            elif operand.IsArg:
                operand_strs.append(f"c[0x0][0x{operand.ArgOffset:x}]")
            elif operand.IsSpecialReg:
                operand_strs.append(operand.Name)
            elif operand.IsImmediate:
                operand_strs.append(hex(operand.ImmediateValue))
            else:
                operand_strs.append(operand.Name if operand.Name else "<??>")

        if self.IsPredicateReg(self.opcodes[0]):
            content = f"{self.opcodes[0]} {'.'.join(self.opcodes[1:])} {' '.join(operand_strs)}"
        else:
            content = f"{'.'.join(self.opcodes)} {' '.join(operand_strs)}"

        return content

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


    def Lift(self, lifter, IRBuilder: ir.IRBuilder, IRRegs, ConstMem):
        Idx = 0
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx += 1
        opcode = self._opcodes[Idx]

        def _get_val(op, name=""):
            if op.IsZeroReg:
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

        if opcode == "MOV":
            dest, src = self._operands[0], self._operands[1]

            # if src.IsArg and IRArgs.get(src.ArgOffset) is None:
            #     print(f"Warning: MOV source argument {src.ArgOffset} is not defined.")
            #     return

            val = _get_val(src, "mov")
            IRRegs[dest.GetIRRegName(lifter)] = val

        elif opcode == "MOV32I":
            dest, src = self._operands[0], self._operands[1]
            if not src.IsImmediate:
                raise UnsupportedInstructionException(f"MOV32I expects immediate, got: {src}")
            val = lifter.ir.Constant(src.GetIRType(lifter), src.ImmediateValue)
            IRRegs[dest.GetIRRegName(lifter)] = val
        
        elif opcode == "IMAD":
            dest, op1, op2, op3 = self._operands[0], self._operands[1], self._operands[2], self._operands[3]
            v1 = _get_val(op1, "imad_lhs")
            v2 = _get_val(op2, "imad_rhs")
            v3 = _get_val(op3, "imad_addend")

            # Handle imad.wide
            if len(self.opcodes) > 1 and self.opcodes[1] == "WIDE":
                v1 = IRBuilder.zext(v1, lifter.ir.IntType(64), "imad_lhs_wide")
                v2 = IRBuilder.zext(v2, lifter.ir.IntType(64), "imad_rhs_wide")
                v3 = IRBuilder.zext(v3, lifter.ir.IntType(64), "imad_addend_wide")

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
            
        elif opcode == "IADD32I":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            idxv = _get_val(op1, "iadd32i_idx")
            base = _get_val(op2, "iadd32i_ptr")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.gep(base, [idxv], "iadd32i")

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

        elif opcode == "SULD":
            dest, ptr = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "suld_addr")
            val = IRBuilder.load(addr, "suld")
            IRRegs[dest.GetIRRegName(lifter)] = val
                
        elif opcode == "STG":
            ptr, val = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "stg_addr")
            v = _get_val(val, "stg_val")
            IRBuilder.store(v, addr)

        elif opcode == "SUST":
            ptr, val = self._operands[0], self._operands[1]
            addr = _get_val(ptr, "sust_addr")
            v = _get_val(val, "sust_val")
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
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = ConstMem[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")
                
                # Add operands
                #IRVal = IRBuilder.mul(Load2, Indices, "mul")

                # Store the value
                #IRBuilder.store(IRVal, IRRes)

        elif opcode == "LD":
             # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "LOP":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "LOP32I":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BFE":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BFI":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "ST":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "SSY":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BRA":
            # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BRK":
             # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif opcode == "IMNMX":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "imnmx_lhs")
            v2 = _get_val(op2, "imnmx_rhs")
            cond = IRBuilder.icmp_signed('>', v1, v2, "imnmx_cmp")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.select(cond, v1, v2, "imnmx_max")
                
        elif self._opcodes[Idx] == "PSETP":
             # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "PBK":
             # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        # elif self._opcodes[Idx] == "@P1" or self._opcodes[Idx] == "@!P1":
        #     return
        elif opcode == "ISET":
            dest, op1, op2 = self._operands[0], self._operands[1], self._operands[2]
            v1 = _get_val(op1, "iset_o1")
            v2 = _get_val(op2, "iset_o2")
            cond = IRBuilder.icmp_signed('==', v1, v2, "iset_cmp")
            IRRegs[dest.GetIRRegName(lifter)] = IRBuilder.zext(cond, ir.IntType(32), "iset_result")
                
        elif self._opcodes[Idx] == "LEA":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            Op3 = self._operands[3]
            
            IRRes = IRRegs[Res.GetIRRegName(lifter)]
            
            if Op1.IsReg:
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                Load1 = IRBuilder.load(IROp1, "lea_op1")
            else:
                print(f"Warning: LEA operand 1 type not fully supported: {Op1}")
                Load1 = lifter.ir.Constant(lifter.ir.IntType(32), 0)
            
            if Op2.IsReg:
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
                Load2 = IRBuilder.load(IROp2, "lea_op2")
            else:
                print(f"Warning: LEA operand 2 type not fully supported: {Op2}")
                Load2 = lifter.ir.Constant(lifter.ir.IntType(32), 0)
            
            if Op3.IsReg:
                if hasattr(Op3, 'Name') and Op3.Name == 'RZ':
                    Load3 = lifter.ir.Constant(lifter.ir.IntType(32), 0)  # RZ is zero register
                else:
                    IROp3 = IRRegs[Op3.GetIRRegName(lifter)]
                    Load3 = IRBuilder.load(IROp3, "lea_op3")
            else:
                print(f"Warning: LEA operand 3 type not fully supported: {Op3}")
                Load3 = lifter.ir.Constant(lifter.ir.IntType(32), 0)
            
            # LEA instruction: compute effective address
            # For LEA.HI, this would typically be base + (index * scale) + offset
            # Simplified implementation: just add the operands
            Temp = IRBuilder.add(Load1, Load2, "lea_temp")
            IRVal = IRBuilder.add(Temp, Load3, "lea_result")
            
            # Store result
            IRBuilder.store(IRVal, IRRes)
                
        elif self._opcodes[Idx] == "CS2R":
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
                elif ValOp.IsZeroReg:
                    IRVal = ir.Constant(ir.IntType(32), 0)
                else:
                    print(f"CS2R: Unknown special register {ValOp}")
                    IRVal = ir.Constant(ir.IntType(32), 0)
                
                IRBuilder.store(IRVal, IRResOp)
                    
        elif self._opcodes[Idx] == "F2I":
            # F2I (Float to Integer conversion)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                Load1 = IRBuilder.load(IROp1, "f2i_input")
                
                # Parse modifiers
                modifiers = self.ParseInstructionModifiers(self._opcodes)
                
                # Determine target type
                target_type = ir.IntType(32)  # default
                if 'type' in modifiers:
                    if modifiers['type'] in ['S16', 'U16']:
                        target_type = ir.IntType(16)
                    elif modifiers['type'] in ['S64', 'U64']:
                        target_type = ir.IntType(64)
                
                # Determine signed vs unsigned
                if 'type' in modifiers and modifiers['type'].startswith('U'):
                    IRVal = IRBuilder.fptoui(Load1, target_type, "f2i_unsigned")
                else:
                    IRVal = IRBuilder.fptosi(Load1, target_type, "f2i_signed")
                
                # If needed, extend to 32-bit for storage
                if target_type.width < 32:
                    if 'type' in modifiers and modifiers['type'].startswith('U'):
                        IRVal = IRBuilder.zext(IRVal, ir.IntType(32), "f2i_zext")
                    else:
                        IRVal = IRBuilder.sext(IRVal, ir.IntType(32), "f2i_sext")
                
                IRBuilder.store(IRVal, IRRes)
                    
        elif self._opcodes[Idx] == "I2F":
            # I2F (Integer to Float conversion)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                Load1 = IRBuilder.load(IROp1, "i2f_input")
                
                # Parse modifiers
                modifiers = self.ParseInstructionModifiers(self._opcodes)
                
                # Determine target float type
                target_type = ir.FloatType()  # F32 default
                if 'float_type' in modifiers:
                    if modifiers['float_type'] == 'F64':
                        target_type = ir.DoubleType()
                    elif modifiers['float_type'] == 'F16':
                        target_type = ir.HalfType()
                
                # Determine signed vs unsigned source
                if 'type' in modifiers and modifiers['type'].startswith('U'):
                    IRVal = IRBuilder.uitofp(Load1, target_type, "i2f_unsigned")
                else:
                    IRVal = IRBuilder.sitofp(Load1, target_type, "i2f_signed")
                
                IRBuilder.store(IRVal, IRRes)
                    
        elif self._opcodes[Idx] == "IABS":
            # IABS (Integer Absolute Value)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                Load1 = IRBuilder.load(IROp1, "iabs_input")
                
                # Parse modifiers for bit width
                modifiers = self.ParseInstructionModifiers(self._opcodes)
                bit_width = 32  # default
                if 'type' in modifiers:
                    if '16' in modifiers['type']:
                        bit_width = 16
                    elif '64' in modifiers['type']:
                        bit_width = 64
                
                # Use appropriate LLVM abs intrinsic
                abs_fn_name = f"llvm.abs.i{bit_width}"
                abs_fn_type = ir.FunctionType(ir.IntType(bit_width), 
                                             [ir.IntType(bit_width), ir.IntType(1)])
                abs_fn = ir.Function(lifter.module, abs_fn_type, abs_fn_name)
                is_poison_on_overflow = ir.Constant(ir.IntType(1), 0)  # false
                
                # Truncate/extend input if needed
                if bit_width != 32:
                    if bit_width < 32:
                        Load1 = IRBuilder.trunc(Load1, ir.IntType(bit_width), "iabs_trunc")
                    else:
                        Load1 = IRBuilder.sext(Load1, ir.IntType(bit_width), "iabs_sext")
                
                IRVal = IRBuilder.call(abs_fn, [Load1, is_poison_on_overflow], "iabs")
                
                # Extend back to 32-bit if needed
                if bit_width < 32:
                    IRVal = IRBuilder.sext(IRVal, ir.IntType(32), "iabs_extend")
                
                IRBuilder.store(IRVal, IRRes)
                    
        elif self._opcodes[Idx] == "LOP3":
            # LOP3 (3-input Logic Operation)
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2] 
            Op3 = self._operands[3]
            
            if Res.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                
                # Handle operands
                if Op1.IsReg:
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                    Load1 = IRBuilder.load(IROp1, "lop3_op1")
                else:
                    Load1 = ir.Constant(ir.IntType(32), 0)
                
                if Op2.IsReg:
                    IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
                    Load2 = IRBuilder.load(IROp2, "lop3_op2")
                else:
                    Load2 = ir.Constant(ir.IntType(32), 0)
                
                if Op3.IsReg:
                    if hasattr(Op3, 'Name') and Op3.Name == 'RZ':
                        Load3 = ir.Constant(ir.IntType(32), 0)
                    else:
                        IROp3 = IRRegs[Op3.GetIRRegName(lifter)]
                        Load3 = IRBuilder.load(IROp3, "lop3_op3")
                else:
                    Load3 = ir.Constant(ir.IntType(32), 0)
                
                # Extract LUT value
                lut_value = self.ExtractLUTFromOperands()
                lut = ir.Constant(ir.IntType(8), lut_value)
                
                # Use NVVM LOP3 intrinsic
                lop3_fn_type = ir.FunctionType(ir.IntType(32),
                                             [ir.IntType(32), ir.IntType(32), 
                                              ir.IntType(32), ir.IntType(8)])
                lop3_fn = ir.Function(lifter.module, lop3_fn_type, "llvm.nvvm.lop3.b32")
                IRVal = IRBuilder.call(lop3_fn, [Load1, Load2, Load3, lut], "lop3")
                IRBuilder.store(IRVal, IRRes)
                    
        elif self._opcodes[Idx] == "MOVM":
            # MOVM (Move Matrix/Vector)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                
                # Parse modifiers
                # MOVM.16.MT88 means 16-bit elements, matrix transpose 8x8
                # For tensor cores, this is moving matrix fragments
                
                # For now, treat as simple register move
                # In a complete implementation, this would handle matrix fragments
                Load1 = IRBuilder.load(IROp1, "movm_input")
                IRBuilder.store(Load1, IRRes)
                
                # Note: proper implementation would use NVVM matrix intrinsics
                print(f"MOVM: Matrix operation simplified to register move")
                    
        elif self._opcodes[Idx] == "MUFU":
            # MUFU (Multi-Function Unit)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                Load1 = IRBuilder.load(IROp1, "mufu_input")
                
                # Parse function type from modifiers
                modifiers = self.ParseInstructionModifiers(self._opcodes)
                func_type = modifiers.get('function', 'RCP')  # default to RCP
                
                # Map to appropriate LLVM intrinsic
                if func_type == 'RCP':
                    # Reciprocal
                    fn_name = "llvm.nvvm.rcp.approx.ftz.f32"
                    fn_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
                elif func_type == 'RSQ':
                    # Reciprocal square root
                    fn_name = "llvm.nvvm.rsqrt.approx.ftz.f32"
                    fn_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
                elif func_type == 'SIN':
                    # Sine
                    fn_name = "llvm.nvvm.sin.approx.ftz.f32"
                    fn_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
                elif func_type == 'COS':
                    # Cosine
                    fn_name = "llvm.nvvm.cos.approx.ftz.f32"
                    fn_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
                elif func_type == 'EX2':
                    # 2^x
                    fn_name = "llvm.nvvm.ex2.approx.ftz.f32"
                    fn_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
                elif func_type == 'LG2':
                    # log2(x)
                    fn_name = "llvm.nvvm.lg2.approx.ftz.f32"
                    fn_type = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
                else:
                    # Unknown function - default to identity
                    print(f"MUFU: Unknown function {func_type}")
                    IRBuilder.store(Load1, IRRes)
                    return
                
                fn = ir.Function(lifter.module, fn_type, fn_name)
                IRVal = IRBuilder.call(fn, [Load1], f"mufu_{func_type.lower()}")
                IRBuilder.store(IRVal, IRRes)
                    
        elif self._opcodes[Idx] == "HMMA":
            # HMMA (Half-precision Matrix Multiply Accumulate)
            # This is a tensor core instruction
            # Format: HMMA.m16n8k8.F32 Rd, Ra, Rb, Rc
            # Where m16n8k8 means 16x8x8 matrix dimensions
            if len(self._operands) >= 4:
                Res = self._operands[0]
                MatA = self._operands[1]
                MatB = self._operands[2]
                MatC = self._operands[3]
                
                if Res.IsReg and MatA.IsReg and MatB.IsReg and MatC.IsReg:
                    IRRes = IRRegs[Res.GetIRRegName(lifter)]
                    IRMatA = IRRegs[MatA.GetIRRegName(lifter)]
                    IRMatB = IRRegs[MatB.GetIRRegName(lifter)]
                    IRMatC = IRRegs[MatC.GetIRRegName(lifter)]
                    
                    # Load matrices
                    LoadA = IRBuilder.load(IRMatA, "hmma_a")
                    LoadB = IRBuilder.load(IRMatB, "hmma_b")
                    LoadC = IRBuilder.load(IRMatC, "hmma_c")
                    
                    # Parse matrix dimensions from instruction
                    # Example: HMMA.1688.F32 -> 16x8x8
                    # The actual tensor core operation would use NVVM intrinsics
                    # like llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16
                    
                    # For now, simplified: D = A*B + C
                    # In reality, this operates on matrix fragments
                    Mul = IRBuilder.mul(LoadA, LoadB, "hmma_mul")
                    Add = IRBuilder.add(Mul, LoadC, "hmma_acc")
                    
                    IRBuilder.store(Add, IRRes)
                    
                    print(f"HMMA: Tensor core operation simplified - proper implementation needs WMMA intrinsics")

        else:
            print("lift instruction: ", self._opcodes[Idx])
            raise UnsupportedInstructionException 

    # Lift branch instruction
    def LiftBranch(self, lifter, IRBuilder, IRRegs, TrueBr, FalseBr, ConstMem):

        def _get_val(op, name=""):
            if op.IsZeroReg:
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

        Idx = 0
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
            
        if self._opcodes[Idx] == "ISETP":
            
            r = _get_val(self._operands[2], "branch_operand_0")

            for i in range(1, len(self.opcodes)):
                temp = _get_val(self._operands[i + 2], f"branch_operand_{i}")
                if self.opcodes[i] == "AND":
                    r = IRBuilder.and_(r, temp, f"branch_and_{i}")
                else:
                    r = IRBuilder.icmp_signed(lifter.GetCmpOp(self.opcodes[i]), r, temp, f"branch_cmp_{i}")
            
            IRBuilder.cbranch(r, TrueBr, FalseBr)
            
    def dump(self):
        print("inst: ", self._id, self._opcodes)
        for operand in self._operands:
            operand.dump()