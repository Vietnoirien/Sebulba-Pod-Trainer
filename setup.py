from setuptools import setup, find_packages

setup(
    name="sebulba_pod_trainer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "wxPython",
    ],
    entry_points={
        'console_scripts': [
            'sebulba=sebulba_pod_trainer.gui.app:main',
        ],
    },
)
