# Special register constants
SR_TID = 'SR_TID'
SR_NTID = 'SR_NTID'
SR_CTAID = 'SR_CTAID'
SR_GRID_DIM = 'SR_GRID_DIM'
SR_LANE = 'SR_LANE'
SR_WARP = 'SR_WARP'
SR_WARPSIZE = 'SR_WARPSIZE'
SR_WARPSZ = 'SR_WARPSZ'
SR_CLOCK = 'SR_CLOCK'
SR_EQMASK = 'SR_EQMASK'
SR_LEMASK = 'SR_LEMASK'
SR_LTMASK = 'SR_LTMASK'
SR_GEMASK = 'SR_GEMASK'
SR_GTMASK = 'SR_GTMASK'
SR_ACTIVEMASK = 'SR_ACTIVEMASK'

class InvalidOperandException(Exception):
    pass

class Operand:
    
    @classmethod
    def Parse(cls, Operand_Content):
        Operand_Content = Operand_Content.lstrip()
        
        Name = Operand_Content
        RegName = None
        IsReg = False
        IsConstMem = False
        IsMemAddr = False
        IsImmediate = False
        Prefix = None
        Suffix = None
        OffsetOrImm = None
        IndexReg = None
        ConstMemBank = None
        
        # For aggregator pass, to allow correct parsing
        # if c[0x1][0x104]:c[0x1][0x100] is given, only parse c[0x1][0x100]
        # Similarly -0x2:-0x1 -> -0x1
        # However, we want to keep register as R2:R1 as it differentiates from R1 or R2
        if ':' in Operand_Content and "0x" in Operand_Content:
            Operand_Content = Operand_Content.split(':')[1]
        
        if Operand_Content.startswith('-'):
            Prefix = '-'
            Operand_Content = Operand_Content[1:]
        elif Operand_Content.startswith('!'):
            Prefix = '!'
            Operand_Content = Operand_Content[1:]
        elif Operand_Content.startswith('~'):
            Prefix = '~'
            Operand_Content = Operand_Content[1:]
        elif Operand_Content.startswith('|'):
            Prefix = '|'
            Operand_Content = Operand_Content[1:-1]
        
        if Operand_Content.startswith('c['):
            ConstMemBank = int(Operand_Content[2:5], 16)
            Operand_Content = Operand_Content[7:-1]
            IsConstMem = True
            
        if Operand_Content.startswith('['):
            Operand_Content = Operand_Content[1:-1]
            IsMemAddr = True

        SubOps = Operand_Content.split('+')
        for SubOperand in SubOps:
            # Suboperand is an immediate(offset)
            if not ("R" in SubOperand or "P" in SubOperand):
                if "0x" in SubOperand:
                    OffsetOrImm = int(SubOperand, 16)
                else: # Try match as decimal
                    try:
                        OffsetOrImm = float(SubOperand)
                    except ValueError:
                        pass
                if Prefix == '-' and OffsetOrImm is not None:
                    OffsetOrImm = -OffsetOrImm
                    Prefix = None
                continue
            
            # Otherwise suboperand is a register
            IsReg = True
            
            # Match suffix(.reuse, .H1, .X4)
            if '.' in SubOperand:
                SubOperand, Suffix = SubOperand.split('.', 1)
            
            # Match prefix(-, !, ||)
            if SubOperand.startswith('-'):
                Prefix = '-'
                SubOperand = SubOperand[1:]
            elif SubOperand.startswith('!'):
                Prefix = '!'
                SubOperand = SubOperand[1:]
            elif SubOperand.startswith('~'):
                Prefix = '~'
                SubOperand = SubOperand[1:]
            elif SubOperand.startswith('|'):
                Prefix = '|'
                SubOperand = SubOperand[1:-1]
            
            
            if RegName is not None:
                IndexReg = SubOperand
            else:
                RegName = SubOperand
                
        if OffsetOrImm is not None and not IsReg and not IsMemAddr and not IsConstMem:
            IsImmediate = True
                    
        return Operand(Name, RegName, IsReg, IsMemAddr, IsConstMem, Prefix, Suffix, OffsetOrImm, IndexReg, ConstMemBank, IsImmediate)
            

    @classmethod
    def fromImmediate(cls, name, value):
        return cls(
            Name=name,
            Reg=None,
            IsReg=False,
            IsMemAddr=False,
            IsConstMem=False,
            Prefix=None,
            Suffix=None,
            OffsetOrImm=value,
            IndexReg=None,
            ConstMemBank=None,
            IsImmediate=True,
        )

    @classmethod
    def fromArg(cls, name, arg_offset, regName=None, prefix=None):
        return cls(
            Name=name,
            Reg=regName,
            IsReg=False,
            IsMemAddr=False,
            IsConstMem=True,
            Prefix=prefix,
            Suffix=None,
            OffsetOrImm=arg_offset,
            IndexReg=None,
            ConstMemBank=0,
            IsImmediate=False,
        )

    @classmethod
    def fromMemAddr(cls, name, reg, suffix=None, offset=0, indexReg=None):
        return cls(
            Name=name,
            Reg=reg,
            IsReg=True,
            IsMemAddr=True,
            IsConstMem=False,
            Prefix=None,
            Suffix=suffix,
            OffsetOrImm=offset,
            IndexReg=indexReg,
            ConstMemBank=None,
            IsImmediate=False,
        )

    @classmethod
    def fromReg(cls, name, reg, suffix=None, prefix=None):
        return cls(
            Name=name,
            Reg=reg,
            IsReg=True,
            IsMemAddr=False,
            IsConstMem=False,
            Prefix=prefix,
            Suffix=suffix,
            OffsetOrImm=None,
            IndexReg=None,
            ConstMemBank=None,
            IsImmediate=False,
        )

    def __init__(self, Name, Reg, IsReg, IsMemAddr, IsConstMem, Prefix=None, Suffix=None, OffsetOrImm=None, IndexReg=None, ConstMemBank=None, IsImmediate=False):
        self.Name = Name
        self.Reg = Reg
        self.IsReg = IsReg
        self.IsMemAddr = IsMemAddr
        self.IsConstMem = bool(IsConstMem)
        self.Prefix = Prefix
        self.Suffix = Suffix
        self.OffsetOrImm = OffsetOrImm
        self.IndexReg = IndexReg
        self.ConstMemBank = ConstMemBank
        self.IsImmediate = bool(IsImmediate)
        self.Skipped = False
        self.TypeDesc = "NOTYPE"
        self.IRType = None
        self.IRRegName = None
        self.DefiningInsts = set()
    @property
    def IsArg(self):
        return bool(self.IsConstMem and self.ConstMemBank == 0)

    @property
    def ArgOffset(self):
        return self.OffsetOrImm if self.IsArg else None

    @property
    def MemAddrOffset(self):
        return self.OffsetOrImm if self.IsMemAddr else None

    @property
    def ImmediateValue(self):
        return self.OffsetOrImm if self.IsImmediate else None

    @ImmediateValue.setter
    def ImmediateValue(self, value):
        self.OffsetOrImm = value
        self.IsImmediate = value is not None

    @property
    def Immediate(self):
        return self.ImmediateValue

    @Immediate.setter
    def Immediate(self, value):
        self.ImmediateValue = value

    @property
    def Offset(self):
        if self.IsMemAddr or self.IsConstMem:
            return self.OffsetOrImm
        return None

    @property
    def IsFloatImmediate(self):
        return bool(self.IsImmediate and self.Name and "0x" not in self.Name.lower())

    @property
    def IsNegativeReg(self):
        return self.Prefix == '-'

    @property
    def IsNotReg(self):
        return self.Prefix in ('!', '~')

    @property
    def IsAbsReg(self):
        return self.Prefix == '|'

    @property
    def IsPredicateReg(self):
        if not self.IsReg:
            return False
        
        if "P" not in self.Reg:
            return False
        
        reg = self.Reg
        reg = reg[1:] if reg.startswith('U') else reg
        reg = reg[1:] if reg.startswith('P') else reg

        return reg[0].isdigit()
    
    @property
    def IsUniformReg(self):
        if not self.IsReg:
            return False
        
        return self.Reg.startswith('U')

    @property
    def IsWritableReg(self):
        if not self.IsReg:
            return False
        if self.IsPT or self.IsRZ or self.IsSpecialReg or self.IsBarrierReg:
            return False
        return True
    
    @property
    def IsBarrierReg(self):
        return self.Reg and self.Reg[0] == 'B'

    @property
    def IsSpecialReg(self):
        return bool(self.Name and (
            self.Name.startswith(SR_TID) or
            self.Name.startswith(SR_NTID) or
            self.Name.startswith(SR_CTAID) or
            self.Name.startswith(SR_GRID_DIM) or
            self.Name.startswith(SR_LANE) or
            self.Name.startswith(SR_WARP) or
            self.Name.startswith(SR_WARPSIZE) or
            self.Name.startswith(SR_WARPSZ) or
            self.Name.startswith(SR_CLOCK) or
            self.Name.startswith(SR_EQMASK) or
            self.Name.startswith(SR_LEMASK) or
            self.Name.startswith(SR_LTMASK) or
            self.Name.startswith(SR_GEMASK) or
            self.Name.startswith(SR_GTMASK) or
            self.Name.startswith(SR_ACTIVEMASK)
        ))

    @property
    def SpecialRegisterAxis(self):
        if not (self.IsSpecialReg and self.Name):
            return None
        if '.' not in self.Name:
            return None
        return self.Name.split('.', 1)[1].split('.', 1)[0].upper()

    def _matches_axis(self, prefix, axis):
        if not (self.Name and self.Name.startswith(prefix)):
            return False
        component = self.SpecialRegisterAxis
        if component is None:
            # Axisless variants default to X for legacy encodings.
            return axis == "X"
        return component == axis

    @property
    def IsThreadIdx(self):
        return self.Name and self.Name.startswith(SR_TID)
    
    @property
    def IsThreadIdxX(self):
        return self._matches_axis(SR_TID, "X")

    @property
    def IsThreadIdxY(self):
        return self._matches_axis(SR_TID, "Y")

    @property
    def IsThreadIdxZ(self):
        return self._matches_axis(SR_TID, "Z")
    
    @property
    def IsBlockDim(self):
        return self.Name and self.Name.startswith(SR_NTID)

    @property
    def IsBlockDimX(self):
        return self._matches_axis(SR_NTID, "X")

    @property
    def IsBlockDimY(self):
        return self._matches_axis(SR_NTID, "Y")

    @property
    def IsBlockDimZ(self):
        return self._matches_axis(SR_NTID, "Z")
    
    @property
    def IsBlockIdx(self):
        return self.Name and self.Name.startswith(SR_CTAID)

    @property
    def IsBlockIdxX(self):
        return self._matches_axis(SR_CTAID, "X")

    @property
    def IsBlockIdxY(self):
        return self._matches_axis(SR_CTAID, "Y")

    @property
    def IsBlockIdxZ(self):
        return self._matches_axis(SR_CTAID, "Z")

    @property
    def IsGridDim(self):
        return self.Name and self.Name.startswith(SR_GRID_DIM)

    @property
    def IsGridDimX(self):
        return self._matches_axis(SR_GRID_DIM, "X")

    @property
    def IsGridDimY(self):
        return self._matches_axis(SR_GRID_DIM, "Y")

    @property
    def IsGridDimZ(self):
        return self._matches_axis(SR_GRID_DIM, "Z")
    
    @property
    def IsLaneId(self):
        return self.Name and self.Name.startswith(SR_LANE)
    
    @property
    def IsWarpId(self):
        return self.Name and self.Name.startswith(SR_WARP)

    @property
    def IsWarpSize(self):
        return self.Name and (
            self.Name.startswith(SR_WARPSIZE) or self.Name.startswith(SR_WARPSZ)
        )

    @property
    def IsLaneMaskEQ(self):
        return self.Name and self.Name.startswith(SR_EQMASK)

    @property
    def IsLaneMaskLE(self):
        return self.Name and self.Name.startswith(SR_LEMASK)

    @property
    def IsLaneMaskLT(self):
        return self.Name and self.Name.startswith(SR_LTMASK)

    @property
    def IsLaneMaskGE(self):
        return self.Name and self.Name.startswith(SR_GEMASK)

    @property
    def IsLaneMaskGT(self):
        return self.Name and self.Name.startswith(SR_GTMASK)

    @property
    def IsActiveMask(self):
        return self.Name and self.Name.startswith(SR_ACTIVEMASK)

    @property
    def IsRZ(self):
        return self.Reg in ["RZ", "SRZ", "URZ"]

    @property
    def IsPT(self):
        return self.Reg in ["PT", "UPT"]
    
    def __str__(self):
        if self.IsMemAddr:
            inner = ""
            if self.Reg:
                reg_text = self.Reg
                if self.Suffix and self.Suffix != "reuse":
                    reg_text = f"{reg_text}.{self.Suffix}"
                inner = reg_text
            if self.IndexReg:
                inner = f"{inner}+{self.IndexReg}" if inner else self.IndexReg
            if self.MemAddrOffset is not None:
                if isinstance(self.MemAddrOffset, str):
                    inner = f"{inner}+{self.MemAddrOffset}" if inner else self.MemAddrOffset
                else:
                    offset = int(self.MemAddrOffset)
                    abs_hex = f"0x{abs(offset):x}"
                    if offset < 0:
                        inner = f"{inner}-{abs_hex}" if inner else f"-{abs_hex}"
                    else:
                        inner = f"{inner}+{abs_hex}" if inner else abs_hex
            return f"[{inner}]" if inner else "[]"
        elif (self.IsPredicateReg or self.IsPT) and self.IsNotReg:
            return f"!{self.Reg}"
        elif self.IsReg:
            if self.IsNotReg:
                if self.IsPredicateReg:
                    s = f"!{self.Reg}"
                else:
                    s = f"~{self.Reg}"
            elif self.IsNegativeReg:
                s =  f"-{self.Reg}"
            elif self.IsAbsReg:
                s =  f"|{self.Reg}|"
            else:
                s = self.Reg

            if self.Suffix and self.Suffix != "reuse":
                return f"{s}.{self.Suffix}"
            else:
                return s
        elif self.IsArg:
            return f"c[0x0][0x{self.ArgOffset:x}]"
        elif self.IsSpecialReg:
            return self.Name
        elif self.IsImmediate:
            if self.IsFloatImmediate:
                return str(self.ImmediateValue)
            return hex(self.ImmediateValue)
        else:
            return self.Name if self.Name else "<??>"
        
    def __repr__(self):
        return self.__str__()

    def SetReg(self, RegName):
        self.Reg = RegName
        self.Name = None
        self.IsReg = True
        self.IsImmediate = False
        # self.Suffix = None
        # self.Prefix = None
        self.Name = self.__str__()

    def Replace(self, other):
        self.Name = other.Name
        self.Reg = other.Reg
        self.IsReg = other.IsReg
        self.IsMemAddr = other.IsMemAddr
        self.IsConstMem = other.IsConstMem
        self.Prefix = other.Prefix
        self.Suffix = other.Suffix
        self.OffsetOrImm = other.OffsetOrImm
        self.IndexReg = other.IndexReg
        self.ConstMemBank = other.ConstMemBank
        self.IsImmediate = other.IsImmediate
        self.Skipped = other.Skipped
        self.TypeDesc = other.TypeDesc
        self.IRType = other.IRType
        self.IRRegName = other.IRRegName
        # Parent don't change 
        # self._Parent = other._Parent

    # Set the type description for operand
    def SetTypeDesc(self, Ty):
        self.TypeDesc = Ty

    # Set the skip flag
    def SetSkip(self):
        self.Skipped = True
        
    # Get the type description
    def GetTypeDesc(self):
        return self.TypeDesc

    def HasTypeDesc(self):
        return self.TypeDesc != "NOTYPE"

    def GetIRType(self, lifter):
        if self.IRType == None:
            self.IRType = lifter.GetIRType(self.TypeDesc)

        return self.IRType

    def GetIRName(self, lifter):
        if self.IsReg:
            return self.Reg + self.TypeDesc
        elif self.IsArg and self.ArgOffset is not None:
            return f"c[0x0][0x{int(self.ArgOffset):x}]" + self.TypeDesc

        return None
    
    def dump(self):
        print("operand: ", self.Name, self.Reg)
    
    def Clone(self):
        cloned = Operand(
            Name=self.Name,
            Reg=self.Reg,
            IsReg=self.IsReg,
            IsMemAddr=self.IsMemAddr,
            IsConstMem=self.IsConstMem,
            Prefix=self.Prefix,
            Suffix=self.Suffix,
            OffsetOrImm=self.OffsetOrImm,
            IndexReg=self.IndexReg,
            ConstMemBank=self.ConstMemBank,
            IsImmediate=self.IsImmediate,
        )
        cloned.Skipped = self.Skipped
        cloned.TypeDesc = self.TypeDesc
        cloned.IRType = self.IRType
        cloned.IRRegName = self.IRRegName
        cloned.DefiningInsts = set(self.DefiningInsts)
        return cloned
