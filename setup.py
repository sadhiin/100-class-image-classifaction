import setuptools
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.0.1"

REPO_NAME = "100(Large)-Class-Image-Classification"
AUTHOOR_NAME = "Sadhin"
SRC_REPO = "large-class-image-classification"
AUTHOR_EMAIL = "sadhin.aiub.cse@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="A large class image classification with Deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/sadhiin/100-class-image-classifaction",
    project_urls={
        "Bug Tracker": f"https://github.com/sadhiin/100-class-image-classifaction/issues",
    },
    install_requires=[
        'numpy==1.24.3',
        'opencv-python==4.8.1.78',
        'tensorflow==2.13',
        'pillow==10.2.0',
        'Flask==3.0.0',
        'Flask-Cors==4.0.0',
        'python-box==7.1.1',
        'ensure==1.0.4'
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9"
)