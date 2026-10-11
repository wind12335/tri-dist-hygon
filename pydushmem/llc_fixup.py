"""Triton hcu pipeline fixup: direct object emission for the final hsaco.

DTK's dushmem bitcode (xfer.bc) selects `v_writelane_b32 vN, sM, m0` inside
dushmemi_transfer_quiet; the DTK AsmParser rejects that instruction as a
constant-bus violation whenever it re-parses .s text, but single-step MC
emission (llc -filetype=obj) accepts it -- the same path DTK's own builds
use, so the encoding is what ships in DTK binaries.

Patch make_amdgcn (stash the llir; it is the only stage that sees it) and
make_hsaco (DTK llc -filetype=obj + ld.lld) instead of triton's
.s -> assemble_amdgcn -> lld chain. Pure python layer; no repo patches.

NB: the backend registry loads hcu/compiler.py via spec_from_file_location
WITHOUT sys.modules registration, so a normal
`from triton.backends.hcu.compiler import HIPBackend` yields a different
module object than the one actually compiling kernels -- patch the
REGISTERED class from triton.backends.backends.
"""
import subprocess
import tempfile

_LLC = "/opt/dtk/llvm/bin/llc"
_LLD = "/opt/dtk/llvm/bin/ld.lld"

_llir_by_kernel = {}
_applied = False


def apply():
    global _applied, _make_amdgcn_orig
    if _applied:
        return
    import triton.backends as backends

    hip_backend = backends.backends["hcu"].compiler
    make_amdgcn_orig = hip_backend.make_amdgcn

    def make_amdgcn(src, metadata, options):
        amdgcn = make_amdgcn_orig(src, metadata, options)  # sets metadata["name"]
        _llir_by_kernel[metadata["name"]] = src
        return amdgcn

    def make_hsaco(src, metadata, options):
        llir = _llir_by_kernel.pop(metadata["name"])
        with tempfile.NamedTemporaryFile(suffix=".ll") as f_ir, \
                tempfile.NamedTemporaryFile(suffix=".o") as f_obj, \
                tempfile.NamedTemporaryFile() as f_out:
            f_ir.write(llir.encode())
            f_ir.flush()
            subprocess.check_call([_LLC, "-mtriple", "amdgcn-amd-amdhsa",
                                   "-mcpu", options.arch, "-filetype=obj",
                                   f_ir.name, "-o", f_obj.name])
            subprocess.check_call([_LLD, "-flavor", "gnu", "-shared",
                                   f_obj.name, "-o", f_out.name])
            with open(f_out.name, "rb") as fd:
                return fd.read()

    hip_backend.make_amdgcn = staticmethod(make_amdgcn)
    hip_backend.make_hsaco = staticmethod(make_hsaco)
    _applied = True
