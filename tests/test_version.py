import hmc_tomography


def test_version():
    """Check if the Versioneer code works."""

    versions = hmc_tomography._version.get_versions()

    for k, i in versions.items():
        print(k, i)
