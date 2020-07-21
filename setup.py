import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vocmax",
    version="0.0.18",
    author="toddkarin",
    author_email="pvtools.lbl@gmail.com",
    description="Calculate the maximum module open circuit voltage to find the maximum solar PV string length",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/toddkarin/vocmax",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy','pandas','pvlib','matplotlib','tqdm','pvfactors'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)