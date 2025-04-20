import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as f:
    license_text = f.read()

requires = [
    "torchao>=0.7.0",
]

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open("src/" + project + '/__init__.py').read())
    return result.group(1)

setuptools.setup(
    name="solow",
    packages=setuptools.find_packages(where="src"),
    version=get_property('__version__', 'solo'),
    package_dir={"solo": "src/solo"},
    author="MTandHJ",
    author_email="congxueric@gmail.com",
    description="SOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    license_files=('LICENSE',),
    python_requires='>=3.8',
    install_requires=requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)