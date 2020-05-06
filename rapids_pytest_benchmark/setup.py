from setuptools import setup

setup(
        name="rapids-pytest-benchmark",
        packages=["rapids_pytest_benchmark"],
        #install_requires=["pytest-benchmark", "asvdb", "pynvml"],
    install_requires=["pytest-benchmark", ],
        # the following makes a plugin available to pytest
        entry_points={"pytest11": ["rapids_benchmark = rapids_pytest_benchmark.plugin"]},
        # custom PyPI classifier for pytest plugins
        classifiers=["Framework :: Pytest"],
    )
