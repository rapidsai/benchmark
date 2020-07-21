# rapids-pytest-benchmark

`rapids-pytest-benchmark` is a plugin to [`pytest`](https://docs.pytest.org/en/latest/contents.html) that extends the functionality of the [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest) plugin by taking advantage of the [hooks exposed by `pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/hooks.html) where applicable, and the plugin classes directly for all other modifications.

Unfortunately, at the time of this writing, none of the hooks provided by `pytest-benchmark` were helpful. However, the longer-term plan is to upstream new `pytest-benchmark` hooks that allow for all the `rapids-pytest-benchmark` functionality to be provided by hooks.  This is greatly preferred as a proper extension technique, since importing and subclassing another plugin's classes directly is a much more fragile technique, since an update to the "parent" plugin could change class APIs and cause breakages. _Note: the conda recipe specifies exactly `pytest-benchmark=3.2.3` to ensure compatibility with the internal class API_

## Installation

`rapids-pytest-benchmark` is available on [conda](https://anaconda.org/rapidsai/rapids-pytest-benchmark) and can be installed with the following command:

```sh
conda install -c rapidsai rapids-pytest-benchmark
```

## How to use `rapids-pytest-benchmark`
- Install it and confirm that the `--benchmark-gpu-*` and `--benchmark-asv-*` options are shown in `pytest --help`
- Add the `gpubenchmark` fixture to your tests/benchmarks, just as one would do with the `benchmark` fixture described [here](https://pytest-benchmark.readthedocs.io/en/latest/usage.html)
- See the help description for the `--benchmark-gpu-*` and `--benchmark-asv-*` options
- Further details are provided [here](../README.md)

## Implementation details: how `rapids-pytest-benchmark` is pulled into a benchmark run
- `rapids-pytest-benchmark` is a standard `pytest` plugin.  See the [`setup.py`](setup.py) file for details on how the `entry_points` specification is used to install the plugin in a way that `pytest` will automatically load it and make it available to users.
- Once loaded, `pytest` looks for various hooks it can call at different points during a `pytest` session in order to allow plugins to extend capabilities.  The list of hooks a `pytest` plugin can call are described [here](https://docs.pytest.org/en/latest/reference.html#hook-reference)
- The hooks `rapids-pytest-benchmark` uses are all defined in the [plugin.py](rapids_pytest_benchmark/plugin.py) file.
- Another key contribution a `pytest` plugin can make it to add new [fixtures](https://docs.pytest.org/en/latest/fixture.html#fixture) a test/benchmark author can use.  `rapids-pytest-benchmark` adds the `gpubenchmark` fixture in [plugin.py](rapids_pytest_benchmark/plugin.py):
```
@pytest.fixture(scope="function")
def gpubenchmark(request, benchmark):
   ...
```
- `rapids-pytest-benchmark` is only used if a benchmark uses the `gpubenchmark` fixture.
- The `gpubenchmark` fixture takes advantage of "chaining" or fixtures calling other fixtures, as described [here](https://docs.pytest.org/en/stable/reference.html#fixtures). `gpubenchmark` calls the `pytest-benchmark` **`benchmark`** fixture to create a standard `pytest-benchmark` instance for all time-based metrics.
- `gpubenchmark` then dynamically wraps the `benchmark` instance in an instance of a `GPUBenchmarkFixture`. The `__getattr__()` method defined in the `GPUBenchmarkFixture` class is written to pass all method calls not overridden by `GPUBenchmarkFixture` on to the standard `benchmark` instance.  This is different than a standard subclassing technique since it allows for an _already instantiated_ instance of a parent class to be overridden by an instance of a subclass.  In fact, the instance that wraps the "parent" instance doesn't even need to be a subclass of it. `GPUBenchmarkFixture` is a subclass just for completeness, and if there's any `isinstance()` calls that expect a standard `Benchmark` instance.
