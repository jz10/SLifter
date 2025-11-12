from __future__ import annotations

from typing import List, Sequence

from transform.transform import SaSSTransform


class Transforms:
    def __init__(self, passes: Sequence[SaSSTransform], *, verbose: bool = False) -> None:
        self.passes: List[SaSSTransform] = list(passes)
        self.verbose = verbose

    def apply(self, module) -> None:
        if self.verbose:
            self._dump_module(module, "Initial module")

        for tranpass in self.passes:
            tranpass.apply(module)
            if self.verbose:
                self._dump_module(module, f"After {tranpass.name}")

    def _dump_module(self, module, title: str) -> None:
        print(title)
        for func in module.functions:
            print(f"Function: {func.name}")
            for block in func.blocks:
                preds = ",".join(bb.addr_content for bb in block._preds)
                succs = ",".join(bb.addr_content for bb in block._succs)
                print(f"  Block: {block.addr_content} from: [{preds}] to: [{succs}]")
                for inst in block.instructions:
                    print(f"    {inst.id}    {inst}")
            print("")
