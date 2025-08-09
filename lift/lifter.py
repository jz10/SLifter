from llvmlite import ir, binding

class Lifter :
    def __init__(self):
        # Initialize LLVM environment
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        self.ir = ir
        self.lift_errors = []
        
    def GetCmpOp(self, Opcode):
        if Opcode == "GE":
            return ">="
        elif Opcode == "EQ":
            return "=="
        elif Opcode == "NE":
            return "!="
        elif Opcode == "LE":
            return "<="
        elif Opcode == "GT":
            return ">"
        elif Opcode == "LT":
            return "<"

        return ""
        
            
    def AddIntrinsics(self, llvm_module):
        # # Create thread idx function
        # FuncTy = self.ir.FunctionType(self.ir.IntType(32), [])
        
        # # Thread index function
        # FuncName = "thread_idx"
        # self.GetThreadIdx = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Block dimension function  
        # FuncName = "block_dim"
        # self.GetBlockDim = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Block index function
        # FuncName = "block_idx" 
        # self.GetBlockIdx = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Lane ID function
        # FuncName = "lane_id"
        # self.GetLaneId = self.ir.Function(llvm_module, FuncTy, FuncName)
        
        # # Warp ID function
        # FuncName = "warp_id"
        # self.GetWarpId = self.ir.Function(llvm_module, FuncTy, FuncName)

        # Constant memory
        ArrayTy =  self.ir.ArrayType(self.ir.IntType(32), 1024)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, ArrayTy, "const_mem")
        
    def LiftModule(self, module, file):
        module.lift(self, file)

    def GetIRType(self, TypeDesc):
        if TypeDesc == "Int32":
            return self.ir.IntType(32)
        elif TypeDesc == "Float32":
            return self.ir.FloatType()
        elif TypeDesc == "Int32_PTR":
            return self.ir.PointerType(self.ir.IntType(32))
        elif TypeDesc == "Float32_PTR":
            return self.ir.PointerType(self.ir.FloatType())
        elif TypeDesc == "Int64":
            return self.ir.IntType(64)
        elif TypeDesc == "Int64_PTR":
            return self.ir.PointerType(self.ir.IntType(64))
        elif TypeDesc == "Int1":
            return self.ir.IntType(1)

        raise ValueError(f"Unknown type: {TypeDesc}")

    def Shutdown(self):
        # Cleanup LLVM environment
        binding.shutdown()
