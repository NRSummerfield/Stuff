from setuptools import setup, find_packages

setup(
    name="Stuff",
    version='0.1',
    description="Tools and functions for seemingly random stuff about image processing and the likes.",
    url="",
    author="Nicholas Summerfield",
    author_email="nsummerfield@wisc.edu",
    packages=['Stuff', 'Stuff.Visualization', 'Stuff.PreProcessing', 'Stuff.BBox'],
    install_requires=[
        "nibabel",
        "numpy",
        "matplotlib",
        "pydicom",
        "scipy",
        "SimpleITK"
    ]
)