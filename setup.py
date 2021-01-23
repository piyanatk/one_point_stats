from setuptools import setup

setup(
    name='one_point_stats',
    version='0.1',
    packages=['one_point_stats',
              'one_point_stats.stats',
              'one_point_stats.utils',
              'one_point_stats.obs',
              'one_point_stats.foreground_filter'],
    url='',
    license='MIT',
    author='Piyanat Kittiwisit',
    author_email='piyanat.kittiwisit@gmail.com',
    description='Tools for 21 cm one-point statistics',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy',
        'scipy',
        'healpy',
        'astropy>3.5',
        'xarray'
    ]
)
