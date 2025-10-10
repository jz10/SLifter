from lift.lifter import Lifter
import ctypes

# Special register constants
SR_TID = 'SR_TID'
SR_NTID = 'SR_NTID'  
SR_CTAID = 'SR_CTAID'
SR_LANE = 'SR_LANE'
SR_WARP = 'SR_WARP'

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
        if Reg.startswith("c["):
            content = Reg.replace("c[0x0][", "").replace("]", "")
            # c[0x0][0x164]:c[0x0][0x160] => use 0x160 as offset
            content = content.split(":")[-1] 
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
    def fromArg(cls, name, arg_offset, negate, not_, abs):
        return cls(Name=name, Reg=None, Suffix=None, ArgOffset=arg_offset, IsReg=False, IsArg=True, IsMemAddr=False, Negate=negate, Not=not_, Abs=abs)

    @classmethod
    def fromMemAddr(cls, name, reg, suffix, offset = 0, uregOffset = None):
        return cls(Name=name, Reg=reg, Suffix=suffix, ArgOffset=offset, IsReg=True, IsArg=False, IsMemAddr=True, URegOffset=uregOffset)

    @classmethod
    def fromReg(cls, name, reg, suffix = None, negate=False, not_=False, abs=False):
        return cls(Name=name, Reg=reg, Suffix=suffix, ArgOffset=None, IsReg=True, IsArg=False, IsMemAddr=False, IsImmediate=False, ImmediateValue=None, Negate=negate, Not=not_, Abs=abs)

    def __init__(self, Name, Reg, Suffix, ArgOffset, IsReg, IsArg, IsMemAddr, IsImmediate=False, ImmediateValue=None, Negate=False, Not=False, Abs=False, URegOffset=None):
        self._Name = Name
        self._Reg = Reg
        self._Suffix = Suffix
        self._ArgOffset = ArgOffset if IsArg else None
        self._IsReg = IsReg
        self._IsArg = IsArg
        self._IsMemAddr = IsMemAddr
        self._IsImmediate = IsImmediate
        self._ImmediateValue = ImmediateValue
        self._MemAddrOffset = ArgOffset if IsMemAddr else None
        self._URegOffset = URegOffset
        self._Skipped = False
        self._NegativeReg = Negate
        self._NotReg = Not
        self._AbsReg = Abs
        self._TypeDesc = "NOTYPE"
        self._IRType = None
        self._IRRegName = None
        self.DefiningInsts = set()

    @property
    def Name(self):
        return self._Name
    
    @property
    def Reg(self):
        return self._Reg

    @property
    def IsReg(self):
        return self._IsReg
    
    @property
    def IsConstMem(self):
        return self._ArgOffset is not None

    @property
    def IsPredicateReg(self):
        if not self.IsReg:
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
    def IsArg(self):
        return self._IsArg
    
    @property
    def IsMemAddr(self):
        return self._IsMemAddr

    @property
    def IsImmediate(self):
        return self._IsImmediate

    @property
    def ImmediateValue(self):
        return self._ImmediateValue
    
    @property
    def IsBarrierReg(self):
        return self.Reg and self.Reg[0] == 'B'

    @property
    def IsSpecialReg(self):
        return self._Name and (self._Name.startswith(SR_TID) or 
                              self._Name.startswith(SR_NTID) or 
                              self._Name.startswith(SR_CTAID) or 
                              self._Name.startswith(SR_LANE) or 
                              self._Name.startswith(SR_WARP))
    
    @property
    def IsNegativeReg(self):
        return self._NegativeReg
    
    @property
    def IsNotReg(self):
        return self._NotReg

    @property
    def IsAbsReg(self):
        return self._AbsReg

    @property
    def IsThreadIdx(self):
        return self._Name and self._Name.startswith(SR_TID)
    
    @property
    def IsBlockDim(self):
        return self._Name and self._Name.startswith(SR_NTID)
    
    @property
    def IsBlockIdx(self):
        return self._Name and self._Name.startswith(SR_CTAID)
    
    @property
    def IsLaneId(self):
        return self._Name and self._Name.startswith(SR_LANE)
    
    @property
    def IsWarpId(self):
        return self._Name and self._Name.startswith(SR_WARP)

    @property
    def IsRZ(self):
        return self.Reg in ["RZ", "SRZ", "URZ"]

    @property
    def IsPT(self):
        return self.Reg in ["PT", "UPT"]
    
    @property
    def ArgOffset(self):
        return self._ArgOffset

    @property
    def TypeDesc(self):
        return self._TypeDesc

    @property
    def Skipped(self):
        return self._Skipped
    
    def __str__(self):
        if self.IsMemAddr:
            reg_text = self._Reg
            if self._Suffix:
                reg_text = f"{self._Reg}.{self._Suffix}"
            if self._URegOffset:
                reg_text = f"{reg_text}+{self._URegOffset}"
            if self._MemAddrOffset:
                return f"[{reg_text}+0x{self._MemAddrOffset:x}]"
            else:
                return f"[{reg_text}]"
        elif (self.IsPredicateReg or self.IsPT) and self._NotReg:
            return f"!{self._Reg}"
        elif self.IsReg:
            if self._NotReg:
                if self.IsPredicateReg:
                    s = f"!{self._Reg}"
                else:
                    s = f"~{self._Reg}"
            elif self._NegativeReg:
                s =  f"-{self._Reg}"
            elif self._AbsReg:
                s =  f"|{self._Reg}|"
            else:
                s = self._Reg

            if self._Suffix:
                return f"{s}.{self._Suffix}"
            else:
                return s
        elif self.IsArg:
            return f"c[0x0][0x{self._ArgOffset:x}]"
        elif self.IsSpecialReg:
            return self._Name
        elif self.IsImmediate:
            return hex(self._ImmediateValue)
        else:
            return self._Name if self._Name else "<??>"
        
    def __repr__(self):
        return self.__str__()

    def SetReg(self, RegName):
        if not self.IsReg:
            raise InvalidOperandException("Cannot set register for non-register operand")
        self._Reg = RegName
        self._Name = RegName

    def Replace(self, other):
        self._Name = other._Name
        self._Reg = other._Reg
        self._Suffix = other._Suffix
        self._ArgOffset = other._ArgOffset
        self._IsReg = other._IsReg
        self._IsArg = other._IsArg
        self._IsMemAddr = other._IsMemAddr
        self._IsImmediate = other._IsImmediate
        self._ImmediateValue = other._ImmediateValue
        self._NegativeReg = other._NegativeReg
        self._NotReg = other._NotReg
        self._AbsReg = other._AbsReg
        self._Skipped = other._Skipped
        self._TypeDesc = other._TypeDesc
        self._IRType = other._IRType
        self._IRRegName = other._IRRegName
        # Parent don't change 
        # self._Parent = other._Parent

    # Set the type description for operand
    def SetTypeDesc(self, Ty):
        self._TypeDesc = Ty

    # Set the skip flag
    def SetSkip(self):
        self._Skipped = True
        
    # Get the type description
    def GetTypeDesc(self):
        return self._TypeDesc

    def HasTypeDesc(self):
        return self._TypeDesc != "NOTYPE"

    def GetIRType(self, lifter):
        if self._IRType == None:
            self._IRType = lifter.GetIRType(self._TypeDesc)

        return self._IRType

    def GetIRName(self, lifter):
        if self.IsReg:
            return self._Reg + self._TypeDesc
        elif self.IsArg:
            return f"c[0x0][0x{self._ArgOffset:x}]" + self._TypeDesc

        return None
    
    def dump(self):
        print("operand: ", self._Name, self._Reg)
    
    def Clone(self):
        return Operand(
            Name=self._Name,
            Reg=self._Reg,
            Suffix=self._Suffix,
            ArgOffset=self._ArgOffset,
            IsReg=self._IsReg,
            IsArg=self._IsArg,
            IsMemAddr=self._IsMemAddr,
            IsImmediate=self._IsImmediate,
            ImmediateValue=self._ImmediateValue,
            Negate=self._NegativeReg,
            Not=self._NotReg,
            Abs=self._AbsReg,
            URegOffset=self._URegOffset
        )