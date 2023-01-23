# Copyright (c) 2020-2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup

import rapids_pytest_benchmark

setup(
        name="rapids-pytest-benchmark",
        version=rapids_pytest_benchmark.__version__,
        packages=["rapids_pytest_benchmark"],
        install_requires=["pytest-benchmark", "asvdb", "pynvml", "rmm"],
        # the following makes a plugin available to pytest
        entry_points={"pytest11": ["rapids_benchmark = rapids_pytest_benchmark.plugin"]},
        # custom PyPI classifier for pytest plugins
        classifiers=["Framework :: Pytest"],
    )
