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

    # def IsBranch(self):
    #     Idx = 0
    #     # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
    #     if self.IsPredicateReg(self._opcodes[Idx]):
    #         Idx = Idx + 1
    #     return self._opcodes[Idx] == "ISETP" 

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
            
    # def ResolveType(self):
    #     if not self.DirectlySolveType():
    #         if not self.PartialSolveType():
    #             print("checking ", self._opcodes[0])
    #             raise UnsupportedOperatorException

    def ResolveType(self):
        if self.DirectlySolveType():
            return True
        
        if self.PartialSolveType():
            return True
        
        return False

    # def ResolveType(self):
    #     if not self.DirectlySolveType():
    #         if not self.PartialSolveType():
    #             print("Failed: ",end="")
    #         else:
    #             print("Success: ",end="")
    #     else:
    #         print("Success: ",end="")
            
    #     print(self._InstContent+" => ",end="")
    #     for operand in self.operands:
    #         if operand.Name:
    #             print(operand.Name,end="")
    #         if operand.TypeDesc:
    #             print("("+operand.TypeDesc+"), ",end="")
    #     print("")

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
        
    # Check and update the use operand's type from the givenn operand
    def CheckAndUpdateUseType(self, Def):
        for i in range(1, len(self._operands)):
            CurrOperand = self._operands[i]
            if CurrOperand.Name == Def.Name:                
                CurrOperand.SetTypeDesc(Def.TypeDesc)
                return True

        return False
    
    # Check and update the def operand's type from the given operands
    def CheckAndUpdateDefType(self, Uses):
        Def = self._operands[0]
        for i in range(len(Uses)):
            CurrUse = Uses[i]
            if CurrUse.IsReg and Def.IsReg and CurrUse.Reg == Def.Reg: # CurrUse.Name == Def.Name:
                Def.SetTypeDesc(CurrUse.TypeDesc)
                return True

        return False
    
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
    
    def DirectlySolveType(self):
        idx = 0
        # skip a leading predicate (P0, !P0, etc.)
        if self.IsPredicateReg(self._opcodes[idx]):
            idx += 1

        op = self._opcodes[idx]

        FLOAT32_OPS = {
            "FADD", "FMUL", "FFMA", "FMA", "FMNMX",
            "FSET", "FSETP", "FCHK", "MUFU",
            "FMIN", "FMAX"
        }

        INT_OPS = {
            "IADD", "ADD", "IADD3", "IMAD", "MADI",
            "XMAD",
            "ISETP", "ISCADD", "IMNMX",
            "AND", "OR", "XOR", "NOT", "LOP", "LOP3",
            "SHL", "SHR", "SHF",
            "ISUB", "LEA", "S2R", "CS2R",
            "IABS", "MOVM",
        }

        if op in FLOAT32_OPS:
            type_desc = "Float32"
        elif op in INT_OPS:
            type_desc = "INT"
        else:
            return False

        for operand in self._operands:
            operand.SetTypeDesc(type_desc)
        return True

    
    def PartialSolveType(self):
        if self._opcodes[0] == "LDG":
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[1].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[1].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == "SULD":
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[1].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[1].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == "STG":
            TypeDesc = self._operands[1].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[0].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[0].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == "SUST":
            TypeDesc = self._operands[1].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[0].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[0].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == 'IADD':
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc:
                self._operands[1].SetTypeDesc("Int32") # The integer offset
                self._operands[2].SetTypeDesc(TypeDesc)
        elif self._opcodes[0] == 'IADD32I':
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc:
                self._operands[1].SetTypeDesc("Int32") # The integer offset
                self._operands[2].SetTypeDesc(TypeDesc)
        else:
            return False

        return True
    def Lift(self, lifter, IRBuilder: ir.IRBuilder, IRRegs, IRArgs):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
            
        if self._opcodes[Idx] == "MOV":
            ResOp = self._operands[0]
            Op1 = self._operands[1]

            if ResOp.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
            else:
                raise UnsupportedInstructionException(f"Unsupported MOV destination operand: {ResOp}")

            if Op1.IsReg:
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IRVal = IRBuilder.load(IROp1, "loadval")
            elif Op1.IsArg:
                if Op1.ArgOffset not in IRArgs:
                    print(f"Skipping MOV to argument {Op1.ArgOffset}, which is below arg offset")
                    return
                
                IRRes = IRArgs[Op1.ArgOffset]
                IRVal = IRBuilder.load(IROp1, "loadval")
            else:
                raise UnsupportedInstructionException(f"Unsupported MOV destination operand: {ResOp}")
            
            try:
                IRBuilder.store(IRVal, IRRes)
            except Exception as e:
                lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "MOV32I":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
        
            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

        
        elif self._opcodes[Idx] == "IMAD":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
        elif self._opcodes[Idx] == "XMAD":
            ResOp = self._operands[1]
            Op1 = self._operands[1]
        elif self._opcodes[Idx] == "EXIT":
            IRBuilder.ret_void()
        elif self._opcodes[Idx] == "FADD":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            if Res.IsReg and Op1.IsReg and Op2.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
            
                # Load operands
                Load1 = IRBuilder.load(IROp1, "load")
                Load2 = IRBuilder.load(IROp2, "load")

                # FADD instruction
                IRVal = IRBuilder.add(Load1, Load2, "fadd")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "ISCADD":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            if Res.IsReg and Op1.IsReg and Op2.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
            
                # Load operands
                Load1 = IRBuilder.load(IROp1, "load")
                Load2 = IRBuilder.load(IROp2, "load")

                # FADD instruction
                IRVal = IRBuilder.add(Load1, Load2, "add")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "IADD3":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            Op3 = self._operands[3]
            
            try:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                
                # Handle first operand
                if Op1.IsReg:
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                    Load1 = IRBuilder.load(IROp1, "load1")
                elif Op1.IsArg:
                    IROp1 = IRArgs[Op1.ArgOffset]
                    Load1 = IRBuilder.load(IROp1, "load1")
                else:
                    # Handle immediate or other operand types
                    print(f"Warning: IADD3 operand 1 type not fully supported: {Op1}")
                    Load1 = lifter.ir.Constant(lifter.ir.IntType(32), 0)  # fallback
                
                # Handle second operand
                if Op2.IsReg:
                    IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
                    Load2 = IRBuilder.load(IROp2, "load2")
                elif Op2.IsArg:
                    IROp2 = IRArgs[Op2.ArgOffset]
                    Load2 = IRBuilder.load(IROp2, "load2")
                else:
                    # Handle immediate or other operand types
                    print(f"Warning: IADD3 operand 2 type not fully supported: {Op2}")
                    Load2 = lifter.ir.Constant(lifter.ir.IntType(32), 0)  # fallback
                
                # Handle third operand
                if Op3.IsReg:
                    IROp3 = IRRegs[Op3.GetIRRegName(lifter)]
                    Load3 = IRBuilder.load(IROp3, "load3")
                elif Op3.IsArg:
                    IROp3 = IRArgs[Op3.ArgOffset]
                    Load3 = IRBuilder.load(IROp3, "load3")
                else:
                    if hasattr(Op3, 'Name') and Op3.Name == 'RZ':
                        Load3 = lifter.ir.Constant(lifter.ir.IntType(32), 0)  # RZ is zero register
                    else:
                        print(f"Warning: IADD3 operand 3 type not fully supported: {Op3}")
                        Load3 = lifter.ir.Constant(lifter.ir.IntType(32), 0)  # fallback

                # IADD3 instruction: add all three operands
                Temp = IRBuilder.add(Load1, Load2, "add_temp")
                IRVal = IRBuilder.add(Temp, Load3, "iadd3")

                # Store result
                IRBuilder.store(IRVal, IRRes)
            except Exception as e:
                lifter.lift_errors.append(e)

        elif self._opcodes[Idx] == "ISUB":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            if Res.IsReg and Op1.IsReg and Op2.IsReg:
                try:
                    IRRes = IRRegs[Res.GetIRRegName(lifter)]
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                    IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
            
                    # Load operands
                    Load1 = IRBuilder.load(IROp1, "load")
                    Load2 = IRBuilder.load(IROp2, "load")

                    # FADD instruction
                    IRVal = IRBuilder.add(Load1, Load2, "sub")

                    # Store result
                    IRBuilder.store(IRVal, IRRes)
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "SHL":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                # Load value
                Load1 = IRBuilder.load(IROp1, "loadval")

                IRVal = IRBuilder.shl(Load1, lifter.ir.Constant(lifter.ir.IntType(32), Op2.ImmediateValue), "add")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "SHR":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                # Load value
                Load1 = IRBuilder.load(IROp1, "loadval")

                # Right shift with immediate value
                IRVal = IRBuilder.lshr(Load1, lifter.ir.Constant(lifter.ir.IntType(32), Op2.ImmediateValue), "shr")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "SHF":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                # Load value
                Load1 = IRBuilder.load(IROp1, "loadval")

                # Funnel shift - for now implement as left shift with immediate
                IRVal = IRBuilder.shl(Load1, lifter.ir.Constant(lifter.ir.IntType(32), Op2.ImmediateValue), "shf")

                # Store result
                IRBuilder.store(IRVal, IRRes)
                
        elif self._opcodes[Idx] == "IADD":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg and ResOp.HasTypeDesc():
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)];
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRArgs[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")

                try:
                    # Add operands
                    IRVal = IRBuilder.gep(Load2, Indices, "addr")

                    # Store the value
                    IRBuilder.store(IRVal, IRRes)
                except Exception as e:
                    lifter.lift_errors.append(e)
            
        elif self._opcodes[Idx] == "IADD32I":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)];
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRArgs[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")
                
                # Add operands
                IRVal = IRBuilder.gep(Load2, Indices, "addr")

                # Store the value
                IRBuilder.store(IRVal, IRRes)
                
        elif self._opcodes[Idx] == "S2R":
            ResOp = self._operands[0]
            ValOp = self._operands[1]
            if ResOp.IsReg:
                try:
                    IRResOp = IRRegs[ResOp.GetIRRegName(lifter)]
                    
                    if ValOp.IsThreadIdx:
                        IRVal = IRBuilder.call(lifter.GetThreadIdx, [], "ThreadIdx")
                    elif ValOp.IsBlockDim:
                        IRVal = IRBuilder.call(lifter.GetBlockDim, [], "BlockDim")
                    elif ValOp.IsBlockIdx:
                        IRVal = IRBuilder.call(lifter.GetBlockIdx, [], "BlockIdx")
                    elif ValOp.IsLaneId:
                        IRVal = IRBuilder.call(lifter.GetLaneId, [], "LaneId")
                    elif ValOp.IsWarpId:
                        IRVal = IRBuilder.call(lifter.GetWarpId, [], "WarpId")
                    else:
                        print(f"S2R: Unknown special register {ValOp.Name}")
                        IRVal = lifter.ir.Constant(lifter.ir.IntType(32), 0)
                    
                    IRBuilder.store(IRVal, IRResOp)
                except Exception as e:
                    lifter.lift_errors.append(e)
                
        elif self._opcodes[Idx] == "LDG":
            PtrOp = self._operands[1]
            ValOp = self._operands[0]
            if PtrOp.IsReg and ValOp.IsReg:
                IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                LoadPtr = IRBuilder.load(IRPtrOp, "loadptr")

                # Load instruction
                IRRes = IRBuilder.load(LoadPtr, "load_inst")

                # Store the result
                IRBuilder.store(IRRes, IRValOp)

        elif self._opcodes[Idx] == "SULD":
            PtrOp = self._operands[1]
            ValOp = self._operands[0]
            #if PtrOp.IsReg and ValOp.IsReg:
                #IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                #IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                #LoadPtr = IRBuilder.load(IRPtrOp, "loadptr")

                # Load instruction
                #IRRes = IRBuilder.load(LoadPtr, "load_inst")

                # Store the result
                #IRBuilder.store(IRRes, IRValOp)
                
        elif self._opcodes[Idx] == "STG":
            PtrOp = self._operands[0]
            ValOp = self._operands[1]
            if PtrOp.IsReg and ValOp.IsReg:
                IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                Addr = IRBuilder.load(IRPtrOp, "loadptr")
                Val = IRBuilder.load(IRValOp, "loadval")

                # Store instruction
                IRBuilder.store(Val, Addr)

        elif self._opcodes[Idx] == "SUST":
            PtrOp = self._operands[0]
            ValOp = self._operands[1]
            #if PtrOp.IsReg and ValOp.IsReg:
                #IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                #IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                #Addr = IRBuilder.load(IRPtrOp, "loadptr")
                #Val = IRBuilder.load(IRValOp, "loadval")

                # Store instruction
                # IRBuilder.store(Val, Addr)
        elif self._opcodes[Idx] == "FMUL":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]

                # Load operands
                Load1 = IRBuilder.load(IROp1, "op1")
                Load2 = IRBuilder.load(IROp2, "op2")

                # Perform multiplication
                IRVal = IRBuilder.mul(Load1, Load2, "mul_result")

                # Store the result
                IRBuilder.store(IRVal, IRRes)
        elif self._opcodes[Idx] == "FFMA":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)];
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRArgs[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")
                
                # Add operands
                #IRVal = IRBuilder.mul(Load2, Indices, "mul")

                # Store the value
                #IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "LD":
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
        elif self._opcodes[Idx] == "IMNMX":
            # IMNMX (Integer Min/Max)
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            
            try:
                if ResOp.IsReg and Op1.IsReg and Op2.IsReg:
                    IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                    IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
                    
                    # Load operands
                    Load1 = IRBuilder.load(IROp1, "imnmx_op1")
                    Load2 = IRBuilder.load(IROp2, "imnmx_op2")
                    
                    # Default to max - should parse modifier
                    Cond = IRBuilder.icmp_signed('>', Load1, Load2, "imnmx_cmp")
                    IRVal = IRBuilder.select(Cond, Load1, Load2, "imnmx_max")
                    
                    # Store result
                    IRBuilder.store(IRVal, IRRes)
            except Exception as e:
                lifter.lift_errors.append(e)
                
        elif self._opcodes[Idx] == "PSETP":
             # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "PBK":
             # TODO: may be sm35 specific?
            ResOp = self._opcodes[0]
        # elif self._opcodes[Idx] == "@P1" or self._opcodes[Idx] == "@!P1":
        #     return
        elif self._opcodes[Idx] == "ISET":
            # ISET (Integer Set)
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            
            try:
                if ResOp.IsReg and Op1.IsReg and Op2.IsReg:
                    IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                    IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
                    
                    # Load operands
                    Load1 = IRBuilder.load(IROp1, "iset_op1")
                    Load2 = IRBuilder.load(IROp2, "iset_op2")
                    
                    # Compare - should parse comparison type
                    Cond = IRBuilder.icmp_signed('==', Load1, Load2, "iset_cmp")
                    
                    # Convert bool to int (0 or 1)
                    IRVal = IRBuilder.zext(Cond, ir.IntType(32), "iset_result")
                    
                    # Store result
                    IRBuilder.store(IRVal, IRRes)
            except Exception as e:
                lifter.lift_errors.append(e)
                
        elif self._opcodes[Idx] == "LEA":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            Op3 = self._operands[3]
            
            try:
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
            except Exception as e:
                lifter.lift_errors.append(e)
                
        elif self._opcodes[Idx] == "CS2R":
            # CS2R (Convert Special Register to Register)
            ResOp = self._operands[0]
            ValOp = self._operands[1]
            if ResOp.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "F2I":
            # F2I (Float to Integer conversion)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "I2F":
            # I2F (Integer to Float conversion)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "IABS":
            # IABS (Integer Absolute Value)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "LOP3":
            # LOP3 (3-input Logic Operation)
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2] 
            Op3 = self._operands[3]
            
            if Res.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "MOVM":
            # MOVM (Move Matrix/Vector)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "MUFU":
            # MUFU (Multi-Function Unit)
            Res = self._operands[0]
            Op1 = self._operands[1]
            if Res.IsReg and Op1.IsReg:
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
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
                
                try:
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
                except Exception as e:
                    lifter.lift_errors.append(e)
        else:
            print("lift instruction: ", self._opcodes[Idx])
            raise UnsupportedInstructionException 
            
    # Lift branch instruction
    def LiftBranch(self, lifter, IRBuilder, IRRegs, IRArgs, TrueBr, FalseBr):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self.opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
            
        if self._opcodes[Idx] == "ISETP":
            Val1Op = self._operands[2]
            Val2Op = self._operands[3]

            # Check register or arguments
            IRVal1 = None
            IRVal2 = None
            if Val1Op.IsReg:
                IRVal1 = IRRegs[Val1Op.GetIRRegName(lifter)]
            elif Val1Op.IsArg:
                IRVal1 = IRArgs[Val1Op.ArgOffset]

            if Val2Op.IsReg:
                IRVal2 = IRRegs[Val2Op.GetIRRegName(lifter)]
            elif Val2Op.IsArg:
                IRVal2 = IRArgs[Val2Op.ArgOffset]

            if not IRVal1 == None and not IRVal2 == None:
                Val1 = IRBuilder.load(IRVal1, "val1")
                Val2 = IRBuilder.load(IRVal2, "val2")

                # Calculate condition
                Cmp = IRBuilder.icmp_signed(lifter.GetCmpOp(self._opcodes[1]), Val1, Val2, "cmp")
                
                # Branch instruction
                IRBuilder.cbranch(Cmp, TrueBr, FalseBr)
            
    def dump(self):
        print("inst: ", self._id, self._opcodes)
        for operand in self._operands:
            operand.dump()