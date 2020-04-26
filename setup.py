from distutils.core import setup, Extension
import numpy as np
module = Extension("myModule", sources = ["myModule.c"])

setup(name="PackageName",
	version = "1.0",
	include_dirs = [np.get_include()],
	ext_modules = [module])