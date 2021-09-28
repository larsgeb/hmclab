def cli():
    import sys
    import hmclab

    print("hmclab cli interface")
    print(f"Version: {hmclab.__version__}")

    # CLI interface
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from hmclab import Tests

        Tests.test_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "himmelblau":
        import hmclab

        target = hmclab.Targets.Himmelblau(annealing=100)
        mass_matrix = hmclab.MassMatrices.Unit(target.dimensions)
        sampler = hmclab.Samplers.HMC(target, mass_matrix, prior)
        filename = "samples_himmelblau.h5"
        sampler.sample(filename, proposals=10000, online_thinning=1, time_step=1.1)
        samples = hmclab.Samples(filename)
        hmclab.Visualization.visualize_2_dimensions(samples, bins=50, show=True)

    exit(0)


if __name__ == "__main__":
    cli()
