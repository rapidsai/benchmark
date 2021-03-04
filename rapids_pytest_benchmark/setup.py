from setuptools import setup

import rapids_pytest_benchmark

setup(
        name="rapids-pytest-benchmark",
        version=rapids_pytest_benchmark.__version__,
        packages=["rapids_pytest_benchmark"],
        install_requires=["pytest-benchmark", "asvdb", "pynvml", "pygal", "rmm"],
        # the following makes a plugin available to pytest
        entry_points={"pytest11": ["rapids_benchmark = rapids_pytest_benchmark.plugin"]},
        # custom PyPI classifier for pytest plugins
        classifiers=["Framework :: Pytest"],
    )
