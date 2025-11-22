SR_TID = "SR_TID"
SR_NTID = "SR_NTID"
SR_CTAID = "SR_CTAID"
SR_GRID_DIM = "SR_GRID_DIM"
SR_LANE = "SR_LANE"
SR_WARP = "SR_WARP"
SR_WARPSIZE = "SR_WARPSIZE"
SR_WARPSZ = "SR_WARPSZ"
SR_CLOCK = "SR_CLOCK"
SR_EQMASK = "SR_EQMASK"
SR_LEMASK = "SR_LEMASK"
SR_LTMASK = "SR_LTMASK"
SR_GEMASK = "SR_GEMASK"
SR_GTMASK = "SR_GTMASK"
SR_ACTIVEMASK = "SR_ACTIVEMASK"


class InvalidOperandException(Exception):
    pass


class Operand:
    @classmethod
    def parse(cls, operand_content):
        operand_content = operand_content.lstrip()

        name = operand_content
        reg_name = None
        is_reg = False
        is_const_mem = False
        is_mem_addr = False
        is_immediate = False
        prefix = None
        suffix = None
        offset_or_imm = None
        index_reg = None
        const_mem_bank = None

        # For aggregator pass, to allow correct parsing
        # if c[0x1][0x104]:c[0x1][0x100] is given, only parse c[0x1][0x100]
        # Similarly -0x2:-0x1 -> -0x1
        # However, we want to keep register as R2:R1 as it differentiates from R1 or R2
        if ":" in operand_content and "0x" in operand_content:
            operand_content = operand_content.split(":")[1]

        if operand_content.startswith("-"):
            prefix = "-"
            operand_content = operand_content[1:]
        elif operand_content.startswith("!"):
            prefix = "!"
            operand_content = operand_content[1:]
        elif operand_content.startswith("~"):
            prefix = "~"
            operand_content = operand_content[1:]
        elif operand_content.startswith("|"):
            prefix = "|"
            operand_content = operand_content[1:-1]

        if operand_content.startswith("c["):
            const_mem_bank = int(operand_content[2:5], 16)
            operand_content = operand_content[7:-1]
            is_const_mem = True

        if operand_content.startswith("["):
            operand_content = operand_content[1:-1]
            is_mem_addr = True

        sub_operands = operand_content.split("+")
        for sub_operand in sub_operands:
            # Suboperand is an immediate(offset)
            if "R" not in sub_operand and "P" not in sub_operand:
                if "0x" in sub_operand:
                    offset_or_imm = int(sub_operand, 16)
                else: # Try match as decimal
                    try:
                        offset_or_imm = float(sub_operand)
                    except ValueError:
                        pass
                if prefix == "-" and offset_or_imm is not None:
                    offset_or_imm = -offset_or_imm
                    prefix = None
                continue
            
            # Otherwise suboperand is a register
            is_reg = True
            
            # Match suffix(.reuse, .H1, .X4)
            if "." in sub_operand:
                sub_operand, suffix = sub_operand.split(".", 1)

            # Match prefix(-, !, ||)
            if sub_operand.startswith("-"):
                prefix = "-"
                sub_operand = sub_operand[1:]
            elif sub_operand.startswith("!"):
                prefix = "!"
                sub_operand = sub_operand[1:]
            elif sub_operand.startswith("~"):
                prefix = "~"
                sub_operand = sub_operand[1:]
            elif sub_operand.startswith("|"):
                prefix = "|"
                sub_operand = sub_operand[1:-1]

            if reg_name is not None:
                index_reg = sub_operand
            else:
                reg_name = sub_operand

        if offset_or_imm is not None and not is_reg and not is_mem_addr and not is_const_mem:
            is_immediate = True

        return cls(
            name=name,
            reg=reg_name,
            is_reg=is_reg,
            is_mem_addr=is_mem_addr,
            is_const_mem=is_const_mem,
            prefix=prefix,
            suffix=suffix,
            offset=offset_or_imm if not is_immediate else None,
            immediate=offset_or_imm if is_immediate else None,
            index_reg=index_reg,
            const_mem_bank=const_mem_bank,
            is_immediate=is_immediate,
        )

    @classmethod
    def from_immediate(cls, name, value):
        return cls(
            name=name,
            reg=None,
            is_reg=False,
            is_mem_addr=False,
            is_const_mem=False,
            prefix=None,
            suffix=None,
            offset=None,
            immediate=value,
            index_reg=None,
            const_mem_bank=None,
            is_immediate=True,
        )

    @classmethod
    def from_cm(cls, name, arg_offset, reg_name=None, prefix=None):
        return cls(
            name=name,
            reg=reg_name,
            is_reg=False,
            is_mem_addr=False,
            is_const_mem=True,
            prefix=prefix,
            suffix=None,
            offset=arg_offset,
            immediate=None,
            index_reg=None,
            const_mem_bank=0,
            is_immediate=False,
        )

    @classmethod
    def from_mem_addr(cls, name, reg, suffix=None, offset=0, index_reg=None):
        return cls(
            name=name,
            reg=reg,
            is_reg=True,
            is_mem_addr=True,
            is_const_mem=False,
            prefix=None,
            suffix=suffix,
            offset=offset,
            immediate=None,
            index_reg=index_reg,
            const_mem_bank=None,
            is_immediate=False,
        )
        
    @classmethod
    def from_reg(cls, name, reg, suffix=None, prefix=None):
        return cls(
            name=name,
            reg=reg,
            is_reg=True,
            is_mem_addr=False,
            is_const_mem=False,
            prefix=prefix,
            suffix=suffix,
            offset=None,
            immediate=None,
            index_reg=None,
            const_mem_bank=None,
            is_immediate=False,
        )

    def __init__(
        self,
        name,
        reg,
        is_reg,
        is_mem_addr,
        is_const_mem,
        is_immediate=False,
        prefix=None,
        suffix=None,
        offset=None,
        immediate=None,
        index_reg=None,
        const_mem_bank=None,
    ):
        self.name = name
        self.reg = reg
        self.is_reg = is_reg
        self.is_mem_addr = is_mem_addr
        self.is_const_mem = is_const_mem
        self.is_immediate = is_immediate or immediate is not None
        self.prefix = prefix
        self.suffix = suffix
        self.offset_value = offset
        self.immediate_value = immediate if self.is_immediate else None
        self.index_reg = index_reg
        self.const_mem_bank = const_mem_bank
        self.type_desc = "NOTYPE"
        self.ir_type = None
        self.ir_reg_name = None
        self.defining_insts = set()

    @property
    def is_arg(self):
        return self.is_const_mem and self.const_mem_bank == 0

    @property
    def offset(self):
        if self.is_mem_addr or self.is_const_mem:
            return self.offset_value
        return None

    @property
    def is_float_immediate(self):
        return self.is_immediate and self.name and "0x" not in self.name.lower()
    
    @property
    def is_negative_reg(self):
        return self.prefix == "-"

    @property
    def is_not_reg(self):
        return self.prefix in ("!", "~")

    @property
    def is_abs_reg(self):
        return self.prefix == "|"

    @property
    def is_predicate_reg(self):
        if not self.is_reg:
            return False

        if "P" not in self.reg:
            return False

        reg = self.reg
        reg = reg[1:] if reg.startswith("U") else reg
        reg = reg[1:] if reg.startswith("P") else reg

        return reg[0].isdigit()

    @property
    def is_uniform_reg(self):
        if not self.is_reg:
            return False

        return self.reg.startswith("U")

    @property
    def is_writable_reg(self):
        if not self.is_reg:
            return False
        if self.is_pt or self.is_rz or self.is_special_reg or self.is_barrier_reg:
            return False
        return True

    @property
    def is_barrier_reg(self):
        return self.reg and self.reg[0] == "B"

    @property
    def is_special_reg(self):
        return bool(
            self.name
            and (
                self.name.startswith(SR_TID)
                or self.name.startswith(SR_NTID)
                or self.name.startswith(SR_CTAID)
                or self.name.startswith(SR_GRID_DIM)
                or self.name.startswith(SR_LANE)
                or self.name.startswith(SR_WARP)
                or self.name.startswith(SR_WARPSIZE)
                or self.name.startswith(SR_WARPSZ)
                or self.name.startswith(SR_CLOCK)
                or self.name.startswith(SR_EQMASK)
                or self.name.startswith(SR_LEMASK)
                or self.name.startswith(SR_LTMASK)
                or self.name.startswith(SR_GEMASK)
                or self.name.startswith(SR_GTMASK)
                or self.name.startswith(SR_ACTIVEMASK)
            )
        )

    @property
    def is_thread_idx(self):
        return self.name and self.name.startswith(SR_TID)

    @property
    def is_thread_idx_x(self):
        return self.is_thread_idx and "X" in self.name

    @property
    def is_thread_idx_y(self):
        return self.is_thread_idx and "Y" in self.name

    @property
    def is_thread_idx_z(self):
        return self.is_thread_idx and "Z" in self.name

    @property
    def is_block_dim(self):
        return self.name and self.name.startswith(SR_NTID)

    @property
    def is_block_dim_x(self):
        return self.is_block_dim and "X" in self.name

    @property
    def is_block_dim_y(self):
        return self.is_block_dim and "Y" in self.name

    @property
    def is_block_dim_z(self):
        return self.is_block_dim and "Z" in self.name
    
    @property
    def is_block_idx(self):
        return self.name and self.name.startswith(SR_CTAID)

    @property
    def is_block_idx_x(self):
        return self.is_block_idx and "X" in self.name

    @property
    def is_block_idx_y(self):
        return self.is_block_idx and "Y" in self.name
    
    @property
    def is_block_idx_z(self):
        return self.is_block_idx and "Z" in self.name

    @property
    def is_grid_dim(self):
        return self.name and self.name.startswith(SR_GRID_DIM)

    @property
    def is_grid_dim_x(self):
        return self.is_grid_dim and "X" in self.name

    @property
    def is_grid_dim_y(self):
        return self.is_grid_dim and "Y" in self.name

    @property
    def is_grid_dim_z(self):
        return self.is_grid_dim and "Z" in self.name

    @property
    def is_lane_id(self):
        return self.name and self.name.startswith(SR_LANE)

    @property
    def is_warp_id(self):
        return self.name and self.name.startswith(SR_WARP)

    @property
    def is_warp_size(self):
        return self.name and (
            self.name.startswith(SR_WARPSIZE) or self.name.startswith(SR_WARPSZ)
        )

    IsWarpSize = is_warp_size

    @property
    def is_lane_mask_eq(self):
        return self.name and self.name.startswith(SR_EQMASK)
    
    @property
    def is_lane_mask_le(self):
        return self.name and self.name.startswith(SR_LEMASK)
    
    @property
    def is_lane_mask_lt(self):
        return self.name and self.name.startswith(SR_LTMASK)

    @property
    def is_lane_mask_ge(self):
        return self.name and self.name.startswith(SR_GEMASK)

    @property
    def is_lane_mask_gt(self):
        return self.name and self.name.startswith(SR_GTMASK)

    @property
    def is_active_mask(self):
        return self.name and self.name.startswith(SR_ACTIVEMASK)

    @property
    def is_rz(self):
        return self.reg in ["RZ", "SRZ", "URZ"]
    
    @property
    def is_pt(self):
        return self.reg in ["PT", "UPT"]

    def __str__(self):
        if self.is_mem_addr:
            inner = ""
            if self.reg:
                reg_text = self.reg
                if self.suffix and self.suffix != "reuse":
                    reg_text = f"{reg_text}.{self.suffix}"
                inner = reg_text
            if self.index_reg:
                inner = f"{inner}+{self.index_reg}" if inner else self.index_reg
            if self.offset_value is not None:
                if isinstance(self.offset_value, str):
                    inner = (
                        f"{inner}+{self.offset_value}"
                        if inner
                        else self.offset_value
                    )
                else:
                    offset = int(self.offset_value)
                    abs_hex = f"0x{abs(offset):x}"
                    if offset < 0:
                        inner = f"{inner}-{abs_hex}" if inner else f"-{abs_hex}"
                    else:
                        inner = f"{inner}+{abs_hex}" if inner else abs_hex
            return f"[{inner}]" if inner else "[]"
        elif (self.is_predicate_reg or self.is_pt) and self.is_not_reg:
            return f"!{self.reg}"
        elif self.is_reg:
            if self.is_not_reg:
                if self.is_predicate_reg:
                    rendered = f"!{self.reg}"
                else:
                    rendered = f"~{self.reg}"
            elif self.is_negative_reg:
                rendered = f"-{self.reg}"
            elif self.is_abs_reg:
                rendered = f"|{self.reg}|"
            else:
                rendered = self.reg

            if self.suffix and self.suffix != "reuse":
                return f"{rendered}.{self.suffix}"
            return rendered
        elif self.is_const_mem:
            return f"c[0x{self.const_mem_bank:x}][0x{self.offset_value:x}]"
        elif self.is_special_reg:
            return self.name
        elif self.is_immediate:
            if self.is_float_immediate:
                return str(self.immediate_value)
            return hex(self.immediate_value)
        return self.name if self.name else "<??>"

    def __repr__(self):
        return self.__str__()

    def set_reg(self, reg_name):
        self.reg = reg_name
        self.name = None
        self.is_reg = True
        self.is_immediate = False
        self.name = str(self)

    def replace(self, other):
        self.name = other.name
        self.reg = other.reg
        self.is_reg = other.is_reg
        self.is_mem_addr = other.is_mem_addr
        self.is_const_mem = other.is_const_mem
        self.prefix = other.prefix
        self.suffix = other.suffix
        self.offset_value = other.offset_value
        self.immediate_value = other.immediate_value
        self.index_reg = other.index_reg
        self.const_mem_bank = other.const_mem_bank
        self.is_immediate = other.is_immediate
        self.type_desc = other.type_desc
        self.ir_type = other.ir_type
        self.ir_reg_name = other.ir_reg_name
        self.defining_insts = other.defining_insts

    def set_type_desc(self, ty):
        self.type_desc = ty

    def get_type_desc(self):
        return self.type_desc

    def has_type_desc(self):
        return self.type_desc != "NOTYPE"

    def get_ir_type(self, lifter):
        return lifter.get_ir_type(self.type_desc)

    def get_ir_name(self, lifter):
        if self.is_reg:
            return self.reg + self.type_desc
        if self.is_const_mem:
            return f"c[0x0{self.const_mem_bank:x}][0x{self.offset_value:x}]" + self.type_desc
        return None

    def dump(self):
        print("operand: ", self.name, self.reg)

    def clone(self):
        cloned = Operand(
            name=self.name,
            reg=self.reg,
            is_reg=self.is_reg,
            is_mem_addr=self.is_mem_addr,
            is_const_mem=self.is_const_mem,
            prefix=self.prefix,
            suffix=self.suffix,
            offset=self.offset_value,
            immediate=self.immediate_value,
            index_reg=self.index_reg,
            const_mem_bank=self.const_mem_bank,
            is_immediate=self.is_immediate,
        )
        cloned.type_desc = self.type_desc
        cloned.ir_type = self.ir_type
        cloned.ir_reg_name = self.ir_reg_name
        cloned.defining_insts = set(self.defining_insts)
        return cloned
