import toml
from setuptools import setup, find_packages

pyproject = toml.load("pyproject.toml")
dependencies = pyproject["tool"]["poetry"]["dependencies"]
extras = pyproject["tool"]["poetry"]["dev-dependencies"]
dependencies.pop("python")
with open("LICENSE") as f:
    mit_license = f.read()
install_requires = [f"{package}{version}" for package, version in dependencies.items()]

setup(
    name='doctable',
    version='0.1.0',
    url='https://github.com/enricoGiga/doctable.git',
    author='enricoGiga',
    author_email='enrico.gigante@gmail.com',
    description='A simple tool to extract tables from pdf and images',
    packages=find_packages(),
    license=mit_license,
    install_requires=install_requires,
    extras_require={"dev": extras},
)
