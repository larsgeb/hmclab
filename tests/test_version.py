import hmclab


def test_version():
    """Check if the Versioneer code works."""

    versions = hmclab._version.get_versions()

    for k, i in versions.items():
        print(k, i)
