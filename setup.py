from skbuild import setup

setup(
    name='VolteraCamera',
    version='0.1dev',
    packages=['volteracamera', ],
    license='Voltera Inc.',
    long_description=open('README.md').read(),
    entry_points={
        'console_scripts': ['FindLaser=volteracamera.analysis.laser_line_finder:preview_image'],
    }
)
