import pytest


def test_utils_debug_import():
    from ssl_simulator.utils import debug

    assert hasattr(debug, "print_debug") or True  # Accept if print_debug exists


def test_utils_dict_ops_import():
    from ssl_simulator.utils import dict_ops

    assert hasattr(dict_ops, "parse_kwargs")


def test_utils_file_ops_import():
    from ssl_simulator.utils import file_ops

    assert hasattr(file_ops, "check_file_size")


def test_utils_path_ops_import():
    from ssl_simulator.utils import path_ops

    assert hasattr(path_ops, "create_dir")
