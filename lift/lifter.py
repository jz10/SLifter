import contextlib
import io
from llvmlite import ir, binding
import llvmlite.binding as llvm
import llvmlite

from transform.transforms import Transforms

class Lifter:

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        pkg_version = getattr(llvmlite, "__version__", None)
        llvm_ver = getattr(
            llvm,
            "llvm_version_info",
            getattr(llvm, "llvm_version", None),
        )

        if self._verbose:
            print("llvmlite package version:", pkg_version)
            if llvm_ver is not None:
                print("LLVM version string:", ".".join(map(str, llvm_ver)))
            print("")

        self.ir = ir
        self.lift_errors = []


    def LiftModule(self, module, outfile):
        passes = self.get_transform_passes()
        transforms = Transforms(passes, verbose=self._verbose)
        if not self._verbose:
            with contextlib.redirect_stdout(io.StringIO()):
                transforms.apply(module)
        else:
            transforms.apply(module)
            
        self.llvm_module = self.ir.Module(module.name)

        self.AddIntrinsics(self.llvm_module)

        for func in module.functions:
            self._lift_function(func, self.llvm_module)
            
        outputIR = str(self.llvm_module)
        
        outputIR = self.postprocess_ir(outputIR)
        
        if self._verbose:
            print(outputIR)
        print(outputIR, file=outfile)
        
    def postprocess_ir(self, ir_code):
        return ir_code
        
    def get_transform_passes(self):
        raise NotImplementedError
