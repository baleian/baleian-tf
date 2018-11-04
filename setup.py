from setuptools import setup, find_packages
 

REQUIRED_PACKAGES = ["tensorflow >= 1.11.0"]
TEST_PACKAGES = []


setup(
      name="baleian-tf",
      version="0.0.1",
      url="https://github.com/baleian/baleian-tf",
      license="MIT",
      author="Beomjoon Lee",
      author_email="baleian90@gmail.com",
      description="Baleian's tensorflow library",
      packages=find_packages(exclude=["examples"]),
      namespace_packages=["baleian"],
      long_description=open("README.md").read(),
      zip_safe=False,
      install_requires=REQUIRED_PACKAGES,
      tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
      )
