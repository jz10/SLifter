import ctypes

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

        # Check if it is an immediate value
        if Operand_Content.startswith('0x') or Operand_Content.startswith('-0x'):
            # Convert to integer, handle pairs as well
            # E.g. -0x1:-0x1=>-0x1, 0x0:0xFF=>0xFF
            ValueStrs = Operand_Content.split(":")
            if len(ValueStrs) == 2:
                ImmediateValue = ((int(ValueStrs[1], 0) & 0xFFFFFFFF) << 32) | (int(ValueStrs[0], 0) & 0xFFFFFFFF)
                ImmediateValue = ctypes.c_int64(ImmediateValue).value
            else:
                ImmediateValue = int(ValueStrs[0], 0)
            Name = Operand_Content
            return Operand.fromImmediate(Name, ImmediateValue)
        else:
            # Try to parse as a floating-point number
            try:
                ImmediateValue = float(Operand_Content)
                Name = Operand_Content
                return Operand.fromImmediate(Name, ImmediateValue)
            except ValueError:
                pass

        # Check if it is a register for address pointer, e.g. [R0]
        if Operand_Content.startswith('[') and Operand_Content.endswith(']'):
            content = Operand_Content[1:-1]

            suffix = None
            if ".X4" in content:
                content = content.replace(".X4", "")
                suffix = "X4"

            if '+' in content:
                items = content.split('+')
                reg = items[0]

                uRegOffset = None
                offset = 0
                if len(items) == 3:
                    uRegOffset = Operand.fromReg(items[2], items[2])
                    offset = int(items[1], base = 16)

                if len(items) == 2:
                    if "UR" in items[1]:
                        uRegOffset = Operand.fromReg(items[1], items[1])
                    else:
                        offset = int(items[1], base = 16)

                if '.' in reg:
                    suffix = reg.split('.')[1]
                    reg = reg.split('.')[0]

                return Operand.fromMemAddr(Operand_Content, reg, suffix, offset, uRegOffset)
            elif '-' in content:
                items = content.split('-')
                reg = items[0]
                offset = -int(items[1], base = 16)
                return Operand.fromMemAddr(Operand_Content, reg, suffix, offset)
            else:
                reg = content
                return Operand.fromMemAddr(Operand_Content, reg, suffix, 0)

        Reg = Operand_Content
        Name = Operand_Content

        # Check for register prefix, including -, ~, or abs '|'
        Negate = Operand_Content.startswith('-')
        Not = Operand_Content.startswith('~') or Operand_Content.startswith('!')
        Abs = Operand_Content.startswith('|') and Operand_Content.endswith('|')
        if Negate or Not:
            Reg = Reg[1:]
        if Abs:
            Reg = Reg[1:-1]

        # Check if it is a function argument or dimension related value
        if Reg.startswith("c[0x0"):
            content = Reg.replace("c[0x0][", "").replace("]", "")
            # c[0x0][0x164]:c[0x0][0x160] => use 0x160 as offset
            content = content.split(":")[-1] 
            
            # c[0x0][R0+0x160]
            regName = None
            if '+' in content:
                regName, content = content.split('+')
                
            offset = int(content, base = 16)

            return Operand.fromArg(Operand_Content, offset, Negate, Not, Abs, regName)
        
        # Check if it is a constant memory address
        if Reg.startswith("c[0x3"):
            content = Reg.replace("c[0x3][", "").replace("]", "")
            offset = int(content, base = 16)
            return Operand.fromArg(Operand_Content, offset, Negate, Not, Abs)

        # Suffix
        Suffix = None
        if not Reg.startswith('SR'):
            Suffix = Reg.split('.')[-1] if '.' in Reg else None
            if Suffix:
                Reg = Reg[:-(len(Suffix) + 1)]

        return Operand.fromReg(Name, Reg, Suffix, Negate, Not, Abs)

    @classmethod
    def fromImmediate(cls, name, value):
        return cls(Name=name, Reg=None, Suffix=None, ArgOffset=None, IsReg=False, IsArg=False, IsMemAddr=False, IsImmediate=True, ImmediateValue=value)

    @classmethod
    def fromArg(cls, name, arg_offset, negate, not_, abs, regName=None):
        return cls(Name=name, Reg=regName, Suffix=None, ArgOffset=arg_offset, IsReg=False, IsArg=True, IsMemAddr=False, Negate=negate, Not=not_, Abs=abs)

    @classmethod
    def fromMemAddr(cls, name, reg, suffix, offset = 0, uregOffset = None):
        return cls(Name=name, Reg=reg, Suffix=suffix, ArgOffset=offset, IsReg=True, IsArg=False, IsMemAddr=True, URegOffset=uregOffset)

    @classmethod
    def fromReg(cls, name, reg, suffix = None, negate=False, not_=False, abs=False):
        return cls(Name=name, Reg=reg, Suffix=suffix, ArgOffset=None, IsReg=True, IsArg=False, IsMemAddr=False, IsImmediate=False, ImmediateValue=None, Negate=negate, Not=not_, Abs=abs)

    def __init__(self, Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate=False, ImmediateValue=None, Negate=False, Not=False, Abs=False, URegOffset=None):
        self.Name = Name
        self.Reg = Reg
        self.Suffix = Suffix
        self.ArgOffset = ArgOffset if IsArg else None
        self.IsReg = IsReg
        self.IsArg = IsArg
        self.IsMemAddr = IsMemAddr
        self.IsImmediate = IsImmediate
        self.ImmediateValue = ImmediateValue
        self.MemAddrOffset = ArgOffset if IsMemAddr else None
        self.URegOffset = URegOffset
        self.Skipped = False
        self.IsNegativeReg = Negate
        self.IsNotReg = Not
        self.IsAbsReg = Abs
        self.TypeDesc = "NOTYPE"
        self.IRType = None
        self.IRRegName = None
        self.IsFloatImmediate = "0x" not in Name if IsImmediate else False
        self.DefiningInsts = set()

    @property
    def IsConstMem(self):
        return self.ArgOffset is not None

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
            reg_text = self.Reg
            if self.Suffix:
                reg_text = f"{self.Reg}.{self.Suffix}"
            if self.URegOffset:
                reg_text = f"{reg_text}+{self.URegOffset}"
            if self.MemAddrOffset:
                return f"[{reg_text}+0x{self.MemAddrOffset:x}]"
            else:
                return f"[{reg_text}]"
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

            if self.Suffix:
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
        if not self.IsReg:
            raise InvalidOperandException("Cannot set register for non-register operand")
        self.Reg = RegName
        self.Name = RegName

    def Replace(self, other):
        self.Name = other.Name
        self.Reg = other.Reg
        self.Suffix = other.Suffix
        self.ArgOffset = other.ArgOffset
        self.IsReg = other.IsReg
        self.IsArg = other.IsArg
        self.IsMemAddr = other.IsMemAddr
        self.IsImmediate = other.IsImmediate
        self.ImmediateValue = other.ImmediateValue
        self.MemAddrOffset = other.MemAddrOffset
        self.URegOffset = other.URegOffset
        self.IsNegativeReg = other.IsNegativeReg
        self.IsNotReg = other.IsNotReg
        self.IsAbsReg = other.IsAbsReg
        self.IsFloatImmediate = other.IsFloatImmediate
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
        elif self.IsArg:
            return f"c[0x0][0x{self.ArgOffset:x}]" + self.TypeDesc

        return None
    
    def dump(self):
        print("operand: ", self.Name, self.Reg)
    
    def Clone(self):
        return Operand(
            Name=self.Name,
            Reg=self.Reg,
            Suffix=self.Suffix,
            ArgOffset=self.ArgOffset if self.IsArg else self.MemAddrOffset,
            IsReg=self.IsReg,
            IsArg=self.IsArg,
            IsMemAddr=self.IsMemAddr,
            IsImmediate=self.IsImmediate,
            ImmediateValue=self.ImmediateValue,
            Negate=self.IsNegativeReg,
            Not=self.IsNotReg,
            Abs=self.IsAbsReg,
            URegOffset=self.URegOffset
        )
