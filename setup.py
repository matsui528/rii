from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension, find_packages
import sys
import setuptools
import re
import subprocess

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = []
    for line in f:
        requirements.append(line.rstrip())

with open('rii/__init__.py') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)



ext_modules = [
    Pybind11Extension(
        'main',
        ['src/main.cpp',
         'src/pqkmeans.cpp'],  # For c++ pqkmeans
        language='c++',
        undef_macros=['NDEBUG'],  # This makes sure assert() works
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']  # https://stackoverflow.com/questions/57456419/errors-building-python-example-from-pybind11-docs-on-macos

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/arch:AVX2'], # '/arch:AVX512' , '/arch:AVX2'
        'unix': [],
    }

    if sys.platform == 'darwin':
        # https://github.com/pybind/pybind11/issues/1604
        # https://github.com/gperftools/gperftools/issues/870#issuecomment-281202369
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-fno-sized-deallocation']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/std:c++17')
            opts.append('/permissive-')
            opts.append('/DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('/openmp') 
            opts.append('/fp:fast') # -Ofast

        if sys.platform not in ['darwin', 'win32']:
            # For linux
            opts.append('-fopenmp')  # For pqk-means.

        if sys.platform not in ['darwin', 'win32']:
            # For linux and mac
            opts.append('-march=native')  # For fast SIMD computation of L2 distance
            opts.append('-mtune=native')  # Do optimization (It seems this doesn't boost, but just in case)
            opts.append('-Ofast')  # This makes the program faster

        for ext in self.extensions:
            ext.extra_compile_args = opts
            if sys.platform not in ['darwin', 'win32']:
                ext.extra_link_args = ['-fopenmp']  # Because of PQk-means

        build_ext.build_extensions(self)

setup(
    name='rii',
    version=version,
    description='Fast and memory-efficient ANN with a subset-search functionality',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Yusuke Matsui',
    author_email='matsui528@gmail.com',
    url='https://github.com/matsui528/rii',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=['pybind11>=2.9'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)
