from collections.abc import Iterator
from pathlib import Path

import pytest

from jarvis.stt import cuda_libraries
from jarvis.stt.cuda_libraries import (
    cuda_library_dirs,
    package_locations,
    register_cuda_libraries,
    reset_registration,
)


@pytest.fixture(autouse=True)
def fresh_registration() -> Iterator[None]:
    reset_registration()
    yield
    reset_registration()


@pytest.fixture
def wheels(tmp_path: Path) -> dict[str, list[str]]:
    """Two fake NVIDIA wheels laid out like site-packages/nvidia/<lib>/bin."""
    layout: dict[str, list[str]] = {}
    for package in ("cublas", "cudnn"):
        root = tmp_path / "nvidia" / package
        (root / "bin").mkdir(parents=True)
        layout[f"nvidia.{package}"] = [str(root)]
    return layout


def test_the_bin_directories_of_both_wheels_are_found(
    wheels: dict[str, list[str]], tmp_path: Path
) -> None:
    dirs = cuda_library_dirs(lambda name: wheels.get(name, []))

    assert dirs == (
        tmp_path / "nvidia" / "cublas" / "bin",
        tmp_path / "nvidia" / "cudnn" / "bin",
    )


def test_windows_registration_adds_dll_directories_and_path(
    wheels: dict[str, list[str]], tmp_path: Path
) -> None:
    added: list[str] = []
    environ = {"PATH": r"C:\Windows"}

    first = register_cuda_libraries(
        platform="win32",
        add_dll_directory=added.append,
        environ=environ,
        locate=lambda name: wheels.get(name, []),
    )
    again = register_cuda_libraries(
        platform="win32",
        add_dll_directory=added.append,
        environ=environ,
        locate=lambda name: wheels.get(name, []),
    )

    cublas = str(tmp_path / "nvidia" / "cublas" / "bin")
    assert added == [cublas, str(tmp_path / "nvidia" / "cudnn" / "bin")]
    assert len(first) == 2
    assert again == ()  # once per process
    assert environ["PATH"].split(";")[-1] == r"C:\Windows"
    assert cublas in environ["PATH"].split(";")


def test_nothing_happens_off_windows(wheels: dict[str, list[str]]) -> None:
    added: list[str] = []

    result = register_cuda_libraries(
        platform="linux",
        add_dll_directory=added.append,
        environ={},
        locate=lambda name: wheels.get(name, []),
    )

    assert result == ()
    assert added == []


def test_missing_wheels_register_nothing() -> None:
    added: list[str] = []
    environ: dict[str, str] = {}

    result = register_cuda_libraries(
        platform="win32", add_dll_directory=added.append, environ=environ, locate=lambda _n: []
    )

    assert result == ()
    assert added == []
    assert environ == {}


def test_package_locations_of_a_missing_package_are_empty() -> None:
    assert package_locations("nvidia_definitely_not_installed.cublas") == ()
    assert package_locations("jarvis.stt") != ()


def test_the_real_extra_is_found_when_installed() -> None:
    dirs = cuda_libraries.cuda_library_dirs()

    assert all(path.name == "bin" for path in dirs)
