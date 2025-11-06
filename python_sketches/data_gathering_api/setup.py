from setuptools import setup

setup(
    name="gdata",
    version="0.1",
    description="A sample Python package",
    author="coding-beagle",
    author_email="nicholasp.teague@gmail.com",
    packages=["gdata"],
    install_requires=["Click", "opencv-python", "numpy", "mediapipe"],
    extras_require={
        "dev": ["pytest", "pytest-cov", "wheel"],
        "test": ["pytest", "pytest-cov"],
    },
    entry_points="""
        [console_scripts]
        gdata=gdata.__main__:cli
    """,
)
