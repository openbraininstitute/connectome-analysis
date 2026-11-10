# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Functions for network analysis on connectomes."""

from connalysis import modelling, network, randomization

__all__ = ["modelling", "network", "randomization"]


def __getattr__(name):
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("connectome-analysis")
        except PackageNotFoundError:  # not installed, e.g. run from a checkout
            return "0+unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
