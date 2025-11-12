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
            
        llvm_module = self.ir.Module(module.name)

        self.AddIntrinsics(llvm_module)

        for func in module.functions:
            self._lift_function(func, llvm_module)

        if self._verbose:
            print(llvm_module)
        print(llvm_module, file=outfile)
        
    def get_transform_passes(self):
        raise NotImplementedError
