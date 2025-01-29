from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


# Read the requirements from requirements.txt
requires = parse_requirements("requirements.txt")

# Dependency links
links = []

setup(
    name="ssl_simulator",
    version="0.1",
    description="",
    author="Swarm Systems Lab",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=requires,
    dependency_links=links,
)
