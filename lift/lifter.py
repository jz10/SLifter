from llvmlite import ir, binding
import llvmlite.binding as llvm
import llvmlite

class Lifter :
    def __init__(self):
        # Initialize LLVM environment
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        pkg_version = getattr(llvmlite, "__version__", None)
        llvm_ver = getattr(llvm, "llvm_version_info",
                        getattr(llvm, "llvm_version", None))

        print("llvmlite package version:", pkg_version)
        if llvm_ver is not None:
            print("LLVM version string:", ".".join(map(str, llvm_ver)))
        print("")

        self.ir = ir
        self.lift_errors = []
        
    def GetCmpOp(self, Opcode):
        if Opcode == "GE" or Opcode == "GEU":
            return ">="
        elif Opcode == "EQ":
            return "=="
        elif Opcode == "NE" or Opcode == "NEU":
            return "!="
        elif Opcode == "LE" or Opcode == "LEU":
            return "<="
        elif Opcode == "GT" or Opcode == "GTU":
            return ">"
        elif Opcode == "LT" or Opcode == "LTU":
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
        ConstArrayTy =  self.ir.ArrayType(self.ir.IntType(8), 4096)
        self.ConstMem = self.ir.GlobalVariable(llvm_module, ConstArrayTy, "const_mem")
        SharedArrayTy =  self.ir.ArrayType(self.ir.IntType(32), 49152)
        self.SharedMem = self.ir.GlobalVariable(llvm_module, SharedArrayTy, "shared_mem")


        # Runtime functions
        self.DeviceFuncs = {}

        # sync threads function
        FuncTy = self.ir.FunctionType(self.ir.VoidType(), [])
        SyncThreads = self.ir.Function(llvm_module, FuncTy, "syncthreads")
        self.DeviceFuncs["syncthreads"] = SyncThreads

        # leader thread function(if (threadIdx.x == 0) *ptr = val)
        FuncTy = self.ir.FunctionType(self.ir.IntType(1), [self.ir.PointerType(self.ir.IntType(32)), self.ir.IntType(32)], False)
        LeaderStore = self.ir.Function(llvm_module, FuncTy, "LeaderStore")
        self.DeviceFuncs["LeaderStore"] = LeaderStore

        # absolute function
        FuncTy = self.ir.FunctionType(self.ir.IntType(32), [self.ir.IntType(32)], False)
        AbsFunc = self.ir.Function(llvm_module, FuncTy, "abs")
        self.DeviceFuncs["abs"] = AbsFunc

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
        elif TypeDesc == "PTR":
            return self.ir.PointerType(self.ir.IntType(32))
        elif TypeDesc == "NOTYPE":
            return self.ir.IntType(32) # Fallback to Int32

        raise ValueError(f"Unknown type: {TypeDesc}")

    def Shutdown(self):
        # Cleanup LLVM environment
        binding.shutdown()
