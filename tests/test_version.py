import hmclab as _hmclab


def test_version():
    """Check if the Versioneer code works."""

    versions = _hmclab._version.get_versions()

    for k, i in versions.items():
        print(k, i)

    print(_hmclab.__version__)
