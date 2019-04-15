def main():
    from hmc_tomography import sampler
    print("Unit test on kinetic energy.")

    sampler = sampler.sampler('tests/kinetic_energy.test.yml')

    sampler.momentum[0] = 4.0
    sampler.momentum[1] = 3.0

    if (12.5 != sampler.kinetic_energy(sampler.momentum)):
        raise Exception("Failed unit test.")

main()
