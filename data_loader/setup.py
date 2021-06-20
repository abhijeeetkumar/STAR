from distutils.core import setup, Extension
import numpy as np

her_utils_module = Extension('HER',
		sources = ['data_loader.c'],
		include_dirs=[np.get_include(), '/home/mdl/amk7371/video_analytics/FFmpeg_x64/include/'],
		extra_compile_args=['-DNDEBUG', '-O3', '-std=c99'],
		extra_link_args=['-L/home/mdl/amk7371/video_analytics/FFmpeg_x64/lib', '-lavutil', '-lavcodec', '-lavformat', '-lswresample', '-lswscale']
)

setup ( name = 'HER',
	version = '0.1',
	description = 'Utils for HEVC Encoded Recognition (HER) training.',
	ext_modules = [ her_utils_module ]
)
