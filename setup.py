from distutils.core import setup, Extension
import numpy as np
module = Extension("myModule", sources = ["myModule.c"],
				extra_compile_args=['-fopenmp'],
				extra_link_args=['-lgomp'])

setup(name="PackageName",
	version = "1.0",
	include_dirs = [np.get_include()],
	ext_modules = [module])