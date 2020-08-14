from src.benchmark import BenchmarkEnums, Benchmark


def test_benchmark_enums():
    for data in BenchmarkEnums:
        Benchmark(data)
    assert False
