# RAPIDS Benchmark

This repo contains tools for benchmarking RAPIDS projects, consisting currently of a plugin to [pytest](https://docs.pytest.org/en/latest) that allows it to run benchmarks to measure execution time and GPU memory usage.

## Contributing Guide

Review [CONTRIBUTING.md](CONTRIBUTING.md) for details about the benchmarking infrastructure relevant to maintaining it (implementation details, design decisions, etc.)

## Benchmarking use cases
### Developer Desktop use case
* Developers write benchmarks using either C++ and the GBench framework, or in Python using the pytest framework with a benchmarking plugin (`rapids-pytest-benchmark`)
* Developers analyze results using the reporting capability of GBench and pytest, or using ASV through the use of the `rapids-pytest-benchmark` `--benchmark-asv-*` options (for python) or a script that converts GBench JSON output for use with ASV (for C++).
### Continuous Benchmarking (CB) - _not fully supported, still WIP_
* Similar in concept to CI, CB runs the repo's benchmark suite (or a subset of it) on a PR to help catch regressions prior to merging
* CB will run the same benchmark code used for the Developer Desktop use case using the same tools (python use `pytest` + `rapids-pytest-benchmark`, C++ uses `GBench` + an output conversion script.)
* CB will update an ASV plot containing only points from the last nightly run and the last release for comparison, then data will be added for each commit within the PR. This will allow a dev to see the affects of their PR changes and give them the opportunity to fix a regression prior to merging.
* CB can be configured to optionally fail a PR if performance degraded beyond an allowable tolerance (configured by the devs)
### Nightly Benchmarking
* A scheduled nightly job will be setup up to run the same benchmarks using the same tools, like the desktop and CB cases above.
* The benchmarks will use the ASV output options (`--benchmark-asv-output-dir`) to generate updates to the nightly ASV database for each repo, which will then be used to render HTML for viewing.

## Writing and running python benchmarks
* Benchmarks for RAPIDS Python APIs can be written in python and run using `pytest` and the `rapids-pytest-benchmark` plugin
* `pytest` is the same tool used for running unit tests, and allows developers to easily transition back and forth between ensuring functional correctness with unit tests, and adequate performance using benchmarks
* `rapids-pytest-benchmark`  is a plugin to `pytest` that extends another plugin named `pytest-benchmark` with GPU measurements and ASV output capabilities.  `pytest-benchmark` is described [here](https://pytest-benchmark.readthedocs.io/en/latest)
* An example of a benchmark running session using `pytest` is below:
```
mymachine:/Projects/cugraph/benchmarks# pytest -v -m small --no-rmm-reinit -k pagerank
========================================================================================================= test session starts ==========================================================================================================
platform linux -- Python 3.6.10, pytest-5.4.3, py-1.8.1, pluggy-0.13.1 -- /opt/conda/envs/rapids/bin/python
cachedir: .pytest_cache
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=3 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=True warmup_iterations=1)
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/Projects/cugraph/benchmarks/.hypothesis/examples')
rapids_pytest_benchmark: 0.0.9
rootdir: /Projects/cugraph/benchmarks, inifile: pytest.ini
plugins: arraydiff-0.3, benchmark-3.2.3, doctestplus-0.7.0, astropy-header-0.1.2, openfiles-0.5.0, remotedata-0.3.1, hypothesis-5.16.0, cov-2.9.0, timeout-1.3.4, rapids-pytest-benchmark-0.0.9
collected 289 items / 287 deselected / 2 selected

bench_algos.py::bench_pagerank[ds=../datasets/csv/directed/cit-Patents.csv,mm=False,pa=False] PASSED                                                                                                                             [ 50%]
bench_algos.py::bench_pagerank[ds=../datasets/csv/undirected/hollywood.csv,mm=False,pa=False] PASSED                                                                                                                             [100%]


---------------------------------------------------------------------------------------------------------- benchmark: 2 tests ---------------------------------------------------------------------------------------------------------
Name (time in ms, mem in bytes)                                                        Min                 Max                Mean            StdDev            Outliers      GPU mem            Rounds            GPU Rounds
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bench_pagerank[ds=../datasets/csv/directed/cit-Patents.csv,mm=False,pa=False]      99.1144 (1.0)      100.3615 (1.0)       99.8562 (1.0)      0.3943 (1.0)           3;0  335,544,320 (2.91)         10          10
bench_pagerank[ds=../datasets/csv/undirected/hollywood.csv,mm=False,pa=False]     171.1847 (1.73)     172.5704 (1.72)     171.9952 (1.72)     0.5118 (1.30)          2;0  115,343,360 (1.0)           6           6
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
================================================================================================== 2 passed, 287 deselected in 15.17s ==================================================================================================
```
The above example demonstrates just a few of the features available:
  * `-m small` - this specifies that only benchmarks using the "small" marker should be run. [Markers](https://docs.pytest.org/en/latest/example/markers.html) allow developers to classify benchmarks and even parameters to benchmarks for easily running subsets of benchmarks interactively.  In this case, benchmarks were written using [parameters](https://docs.pytest.org/en/latest/parametrize.html), and the parameters have markers. These benchmarks have a parameter to define which dataset they should read, and in this case, those marked with the "small" marker are the only ones used for the benchmark runs.
  * `--no-rmm-reinit` - this is a custom option just for these benchmarks.  `pytest` allows users to define their own options for special cases using the [`conftest.py` file](https://docs.pytest.org/en/stable/writing_plugins.html#conftest-py-plugins) and the [`pytest_addoption` API](https://docs.pytest.org/en/stable/writing_plugins.html?highlight=pytest_addoption#using-hooks-in-pytest-addoption)
  * `-k pagerank` - the [`-k`](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests) pytest option allows a user to filter the tests (benchmarks) run to those that match a pattern, in this case, the benchmark names must contain the string "pagerank".

* `rapids-pytest-benchmark` specifically adds these features to `pytest-benchmark`:
  * The `gpubenchmark` fixture.  This is an extension of the `benchmark` fixture provided by `pytest-benchmark`. A developer simply replaces `benchmark` (described [here](https://pytest-benchmark.readthedocs.io/en/latest/usage.html)) with `gpubenchmark` to use the added features.
  * The following CLI options:
```
  --benchmark-gpu-device=GPU_DEVICENO
                        GPU device number to observe for GPU metrics.
  --benchmark-gpu-max-rounds=BENCHMARK_GPU_MAX_ROUNDS
                        Maximum number of rounds to run the test/benchmark
                        during the GPU measurement phase. If not provided, will
                        run the same number of rounds performed for the runtime
                        measurement.
  --benchmark-gpu-disable
                        Do not perform GPU measurements when using the
                        gpubenchmark fixture, only perform runtime measurements.
  --benchmark-asv-output-dir=ASV_DB_DIR
                        ASV "database" directory to update with benchmark
                        results.
  --benchmark-asv-metadata=ASV_DB_METADATA
                        Metadata to be included in the ASV report. For example:
                        "machineName=my_machine2000, gpuType=FastGPU3,
                        arch=x86_64". If not provided, best-guess values will be
                        derived from the environment. Valid metadata is:
                        "machineName", "cudaVer", "osType", "pythonVer",
                        "commitRepo", "commitBranch", "commitHash",
                        "commitTime", "gpuType", "cpuType", "arch", "ram",
                        "gpuRam"
```
  * The report pytest-benchmark prints to the console has also been updated to include the GPU memory usage and the number of GPU benchmark rounds run when a developer uses the `gpubenchmark` fixture, as shown above in the example (`GPU mem` and `GPU Rounds`).

* A common pattern with both unit tests and (now) benchmarks is to define a standard set initial [`pytest.ini`](https://docs.pytest.org/en/latest/customize.html#configuration-file-formats), something similar to the following:
```
[pytest]
addopts =
          --benchmark-warmup=on
          --benchmark-warmup-iterations=1
          --benchmark-min-rounds=3
          --benchmark-columns="min, max, mean, stddev, outliers, rounds"

markers =
          ETL: benchmarks for ETL steps
          small: small datasets
          directed: directed datasets
          undirected: undirected datasets

python_classes =
                 Bench*
                 Test*

python_files =
                 bench_*
                 test_*

python_functions =
                   bench_*
                   test_*
```
The above example adds a specific set of options that a particular project may always want, registers the markers used by the benchmarks (markers should be [registered to prevent a warning](https://docs.pytest.org/en/latest/example/markers.html#registering-markers)), then defines the pattern pytest should match for class names, file names, and function names. Here it's common to have pytest discover both benchmarks (defined here to have a `bench` prefix) and tests (`test` prefix) to allow users to run both in a single run.

Details about writing benchmarks using `pytest-benchmark` (which are 100% applicable to `rapids-pytest-benchmark` if the `gpubenchmark` fixture was used instead) can be found [here](https://pytest-benchmark.readthedocs.io/en/latest/usage.html), and a simple example of a benchmark using the `rapids-pytest-benchmark` features is shown below.
`bench_demo.py`
```
import time
import pytest

@pytest.mark.parametrize("paramA", [0, 2, 5, 9])
def bench_demo(gpubenchmark, paramA):
    # Note: this does not use the GPU at all, so mem usage should be 0
    gpubenchmark(time.sleep, (paramA * 0.1))
```
This file is in the same directory as other benchmarks, so the run can be limited to only the benchmark here using `-k`:
```
(rapids) root@f078ef9f2198:/Projects/cugraph/benchmarks# pytest -k demo --benchmark-gpu-max-rounds=1
========================================================= test session starts ==========================================================
platform linux -- Python 3.6.10, pytest-5.4.3, py-1.8.1, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=3 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=True warmup_iterations=1)
rapids_pytest_benchmark: 0.0.9
rootdir: /Projects/cugraph/benchmarks, inifile: pytest.ini
plugins: arraydiff-0.3, benchmark-3.2.3, doctestplus-0.7.0, astropy-header-0.1.2, openfiles-0.5.0, remotedata-0.3.1, hypothesis-5.16.0, cov-2.9.0, timeout-1.3.4, rapids-pytest-benchmark-0.0.9
collected 293 items / 289 deselected / 4 selected

bench_demo.py ....                                                                                                               [100%]


------------------------------------------------------------------------------------- benchmark: 4 tests -----------------------------------------------------------------------------------------------
Name (time in ns, mem in bytes)                  Min                         Max                        Mean                 StdDev            Outliers  GPU mem            Rounds            GPU Rounds
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bench_demo[0]                               782.3110 (1.0)            2,190.8432 (1.0)              789.0240 (1.0)          12.3101 (1.0)      453;1739        0 (1.0)      126561           1
bench_demo[2]                       200,284,559.2797 (>1000.0)  200,347,900.3906 (>1000.0)  200,329,241.1566 (>1000.0)  26,022.0129 (>1000.0)       1;0        0 (1.0)           5           1
bench_demo[5]                       500,606,104.7316 (>1000.0)  500,676,967.2036 (>1000.0)  500,636,843.3436 (>1000.0)  36,351.5426 (>1000.0)       1;0        0 (1.0)           3           1
bench_demo[9]                       901,069,939.1365 (>1000.0)  901,218,764.4839 (>1000.0)  901,159,526.1594 (>1000.0)  78,917.8600 (>1000.0)       1;0        0 (1.0)           3           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
================================================== 4 passed, 289 deselected in 17.73s ==================================================
```
Below are some important points about this run:
* Since the `-v` option is not used, the compact, abbreviated output is generated using a single `.` for each run (4 in this case)
```
bench_demo.py ....
```
* Notice the time units are in nanoseconds. This was used since the fastest runs were too fast to display in ms or even us (benchmarking the sleep of 0 seconds)
* The `pytest-benchmark` defaults are shown in the output, in particular, `min_rounds=3` and `max_time=1.0` are of interest.
  * `min_rounds` is the minimum number of times the code being benchmarked will be run in order to compute meaningful stats (min, max, mean, std.dev., etc.). Since this is a minimum, pytest-benchmark will often run (many) more rounds than the minimum.
  * `max_time` is used to help determine how many rounds can be run by providing a maximum time, in seconds, for each test/benchmark to run as many rounds as possible in that duration.
* The `--benchmark-gpu-max-rounds=1` option had to be specified. By default, `rapids-pytest-benchmark` will run as many rounds for the separate GPU measurements as were performed by `pytest-benchmark` for the time measurements. Unfortunately, obtaining GPU measurements are very expensive, and much slower than just looking at a timer before and after a call. Because the first parameter was 0, which was a benchmark on the call `time.sleep(0)`, `pytest-benchmark` was able to get 126561 rounds in during the 1.0 second `max_time` duration. Performing 126561 iterations of GPU measurements takes a very long time, so the `--benchmark-gpu-max-rounds=1` option was given to limit the GPU measurments to just 1 round, which is shown in the report. Limiting GPU rounds to a small number is usually aceptable because 1) any warmup rounds that influence GPU measurements were done during the time measurement rounds (which all run to completion before GPU measurements are done), 2) GPU measurements (for memory) are not subject to jitter like time measurements are, so in other words, running the same code will allocate the same number of bytes each time no matter how many times it's run. The reason someone might want to do >1 round at all for GPU measurements is the current GPU measuring code uses a polling technique, which could miss spikes in memory usage (and this becomes much more common the faster the algorithms being run are), and running multiple times helps catch spikes that may have been missed in a prior run.
  * Notes:
    * A future version of `rapids-pytest-benchmark` will use RMM's logging feature to record memory alloc/free transactions for an accurate memory usage measurement that isn't susceptible to missing spikes.
    * A common option to add to `pytest.ini` is `--benchmark-gpu-max-rounds=3`. Since this is a maximum, the number of rounds could be even lower if the algo being benchmarked is slow, and 3 provides a reasonable number of rounds to catch spikes for faster algos.
* As the args to the benchmarked function get larger, we can see the `min_rounds` coming into play more. For a benchmark of `time.sleep(.5)` and `time.sleep(.9)`, which should only allow for 2 and 1 rounds respectively for a `max_time` of 1.0, the `min_rounds` forced 3 runs for better averaging.


## Writing and running C++ benchmarks using gbench
TBD

## Using asvdb from python and the command line
[`asvdb`](https://github.com/rapidsai/asvdb) is a library and command-line utility for reading and writing benchmark results from/to an ASV "database" as described [here](https://asv.readthedocs.io/en/stable/dev.html?highlight=%24results_dir#benchmark-suite-layout-and-file-formats).
* `asvdb` is a key component in the benchmark infrastructure suite in that it is the destination for benchmark results measured by the developer's benchmark code, and the source of data for the benchmarking report tools (in this case just ASV).
* Several examples for both reading and writing a database using the CLI and the API are available [here](https://github.com/rapidsai/asvdb/blob/main/README.md)


## Benchmarking old commits
* It's highly likely that a nightly benchmark run will not be run for a merge commit where the actual regression was introduced. At the moment, the nightly benchmark runs will run on _the last merge commit of the day_, and while the code may contain the regression, the commit that was benchmarked may be the commit to examine when looking for it (it may be in another merge commit that happened earlier in the day, between the current benchmark run and the run from the day before).
* Below is a pseudo-script written as part of benchmarking a series of old commits used to find a regression.  This process illustrates some (hopefully uncommon) scenarios that actually happened, which can greatly complicate the process. The script captures a procedure run in a RAPIDS `devel` docker container:
```
# uninstall rmm cudf cugraph
#  If installed via a local from-source build, use pip and manually remove C++ libs, else use conda
pip uninstall -y rmm cudf dask-cudf cugraph
rm -rf /opt/conda/envs/rapids/include/libcudf
find /opt/conda -type f -name "librmm*" -exec rm -f {} \;
find /opt/conda -type f -name "libcudf*" -exec rm -f {} \;
find /opt/conda -type f -name "libcugraph*" -exec rm -f {} \;
#conda remove -y librmm rmm libcudf cudf dask-cudf libcugraph cugraph

# confirm packages uninstalled with conda list, uninstall again if still there (pip uninstall sometimes needs to be run >once for some reason)
conda list rmm; conda list cudf; conda list cugraph

# install numba=0.48 since older cudf versions being used here need it
conda install -y numba=0.48

# (optional) clone rmm, cudf, cugraph in a separate location if you don't want to modify your working copies (recommended to ensure we're starting with a clean set of sources with no artifacts)
git clone https://github.com/rapidsai/rmm
git clone https://github.com/rapidsai/cudf
git clone https://github.com/rapidsai/cugraph

# copy benchmarks dir from current cugraph for use later in older cugraph
cp -r cugraph/benchmarks /tmp

########################################

# set RMM to old version: 63ebb53bf21a58b98b4596f7b49a46d1d821b05d
#cd <rmm repo>
git reset --hard 63ebb53bf21a58b98b4596f7b49a46d1d821b05d

# install submodules
git submodule update --init --remote --recursive

# confirm the right version (Apr 7)
git log -n1

# build and install RMM
./build.sh

########################################

# set cudf to pre-regression version: 12bd707224680a759e4b274f9ce4013216bf3c1f
#cd <cudf repo>
git reset --hard 12bd707224680a759e4b274f9ce4013216bf3c1f

# install submodules
git submodule update --init --remote --recursive

# confirm the right version (Apr 15)
git log -n1

# build and install cudf
./build.sh

########################################

# set cugraph to version old enough to support old cudf version: 95b80b40b25b733f846da49f821951e3026e9588
#cd <cugraph repo>
git reset --hard 95b80b40b25b733f846da49f821951e3026e9588

# cugraph has no git submodules

# confirm the right version (Apr 16)
git log -n1

# build and install cugraph
./build.sh

########################################

# install benchmark tools and datasets
conda install -c rlratzel -y rapids-pytest-benchmark

# get datasets
#cd <cugraph repo>
cd datasets
mkdir csv
cd csv
wget https://rapidsai-data.s3.us-east-2.amazonaws.com/cugraph/benchmark/benchmark_csv_data.tgz
tar -zxf benchmark_csv_data.tgz && rm benchmark_csv_data.tgz

# copy benchmarks to cugraph
#cd <cugraph repo>
cp -r /tmp/benchmarks .

# verify cudf in PYTHONPATH is correct version (look for commit hash in version)
python -c "import cudf; print(cudf.__version__)"

# run benchmarks
cd benchmarks
pytest -v -m small --benchmark-autosave --no-rmm-reinit -k "not force_atlas2 and not betweenness_centrality"

# confirm that these results are "fast" - on my machine, BFS mean time was ~30ms

########################################

# uninstall cudf
pip uninstall -y cudf dask-cudf
rm -rf /opt/conda/envs/rapids/include/libcudf
find /opt/conda -type f -name "libcudf*" -exec rm -f {} \;
#conda remove -y libcudf cudf dask-cudf

# set cudf to version of regression: 4009501328166b109a73a0a9077df513186ffc2a
#cd <cudf repo>
git reset --hard 4009501328166b109a73a0a9077df513186ffc2a

# confirm the right version (Apr 15 - Merge pull request #4883 from rgsl888prabhu/4862_getitem_setitem_in_series)
git log -n1

# CLEAN and build and install cudf
./build.sh clean
./build.sh

# verify cudf in PYTHONPATH is correct version (look for commit hash in version)
python -c "import cudf; print(cudf.__version__)"

# run benchmarks
#cd <cugraph repo>/benchmarks
pytest -v -m small --benchmark-autosave --no-rmm-reinit -k "not force_atlas2 and not betweenness_centrality" --benchmark-compare --benchmark-group-by=fullname

# confirm that these results are "slow" - on my machine, BFS mean time was ~75ms, GPU mem used was ~3.5x more
#-------------------------------------------------------------------------------------- benchmark 'bench_algos.py::bench_bfs[ds=../datasets/csv/directed/cit-Patents.csv]': 2 tests ---------------------------------------------------------------------------------------
#Name (time in ms, mem in bytes)                                               Min                Max               Mean            StdDev             Median               IQR            Outliers      OPS                GPU mem            Rounds            Iterations
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#bench_bfs[ds=../datasets/csv/directed/cit-Patents.csv] (0001_95b80b4)     27.3090 (1.0)      39.1467 (1.0)      29.5639 (1.0)      2.9815 (1.0)      28.4831 (1.0)      0.8261 (1.0)           5;6  33.8250 (1.0)      117,440,512 (1.0)          34           1
#bench_bfs[ds=../datasets/csv/directed/cit-Patents.csv] (NOW)              70.0455 (2.56)     83.7894 (2.14)     75.5794 (2.56)     3.7335 (1.25)     76.3104 (2.68)     5.2627 (6.37)          5;0  13.2311 (0.39)     432,013,312 (3.68)         15           1
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```