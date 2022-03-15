import versioneer
from setuptools import find_packages, setup

DISTNAME = "cog_datasets"
LICENSE = "MIT"
AUTHOR = "Ryan Soklaski"
AUTHOR_EMAIL = "rsoklaski@gmail.com"
URL = "https://github.com/rsokl/cog_datasets"

INSTALL_REQUIRES = [
    "numpy >= 1.19", "matplotlib >= 3.0.0"
]

DESCRIPTION = "Save machine learning data sets to a common location."
LONG_DESCRIPTION = """
Save machine learning data sets to a common location, and load them without
having to specify a path.

All datasets will be saved to `datasets.path`. By default, this will be point to
~/datasets. You can update this path via `datasets.set_path`.
"""


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    install_requires=INSTALL_REQUIRES,
    url=URL,
    download_url=f"{URL}/tarball/v" + versioneer.get_version(),
    python_requires=">=3.7",
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
)