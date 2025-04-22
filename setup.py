from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def main():
    reqs = [str(req) for req in parse_requirements(open('requirements.txt'))]

    setup(
        name='Metals',
        version='1.0.1',
        author='Kovylyaev',
        description='Carbon regr in metals micros',
        packages=find_packages(),
        python_requires='>=3.9',
        url="https://github.com/Kovylyaev/Metals",
        install_requires=reqs,
    )


if __name__ == '__main__':
    main()