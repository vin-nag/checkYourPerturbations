import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Check Your Perturbations",
    version="0.0.1",
    author="Vineel Nagisetty, Laura Graves, and Joseph Scott",
    author_email="vineel.nagisetty@uwaterloo.ca",
    description="A tool to compare non-gradient based adversarial example generation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vin-nag/checkYourPerturbations",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
