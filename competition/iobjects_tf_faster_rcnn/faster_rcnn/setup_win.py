import numpy as np
import os

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

nvcc_bin = 'nvcc.exe'
lib_dir = 'lib/x64'


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    # self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    # default_compiler_so = self.spawn
    # default_compiler_so = self.rc
    super = self.compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None,
                extra_postargs=None, depends=None):
        postfix = os.path.splitext(sources[0])[1]

        if postfix == '.cu':
            # use the cuda for .cu files
            # self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        return super(sources, output_dir, macros, include_dirs, debug, extra_preargs, postargs, depends)
        # reset the default compiler_so, which we might have changed for cuda
        # self.rc = default_compiler_so

    # inject our redefined _compile method into the class
    self.compile = compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    # unix _compile: obj, src, ext, cc_args, extra_postargs, pp_opts
    Extension(
        "utils.cython_bbox",
        sources=["utils\\bbox.pyx"],
        # define_macros={'/LD'},
        # extra_compile_args={'gcc': ['/link', '/DLL', '/OUT:cython_bbox.dll']},
        # extra_compile_args={'gcc': ['/LD']},
        extra_compile_args={'gcc': []},
        include_dirs=[numpy_include]
    ),
]
# ./setup_win.py build_ext --inplace
setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)