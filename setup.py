import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="qusource",
  version="0.0.4",
  author="Xiangrui Meng",
  author_email="mxr@mail.ustc.edu.cn",
  description="Quantum circuit simulator",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/Meng-Xiang-Rui/Qusource",
  packages=setuptools.find_packages(),
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',
  install_requires=[
      'numpy',
  ],
)