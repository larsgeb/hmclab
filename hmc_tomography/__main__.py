if __name__ == "__main__":
    import sys

    print("hmc_tomography cli interface")

    # CLI interface for tests
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from hmc_tomography import Tests

        Tests.test_all()

    exit(0)
