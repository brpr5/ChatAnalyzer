import setuptools

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chat_analyzer", # Replace with your own username
    version="0.0.1",
    author="Example Author",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brpr5/ChatAnalyzer",
    packages=setuptools.find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires
)
