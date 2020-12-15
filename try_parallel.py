import hmc_tomography
import matplotlib.pyplot as plt

chains = 3
samplers = [hmc_tomography.Samplers.RWMH() for i in range(chains)]
posteriors = [
    hmc_tomography.Distributions.Himmelblau(temperature=temp) for temp in [100, 10, 1]
]
filenames = [f"samples{i}.h5" for i in range(chains)]
hmc_tomography.Samplers.ParallelSampleSMP(
    samplers, filenames, posteriors, 100000, exchange_interval=100
)


samples_objs = [hmc_tomography.Samples(filename) for filename in filenames]
samples = [so.numpy for so in samples_objs]
for so in samples_objs:
    so.close()
    del so

print("\x1b[2K\r")

plt.figure(figsize=(8, 4))

for i in range(chains):
    plt.subplot(1, chains, int(i + 1), aspect="equal")
    plt.hist2d(
        samples[i][0, :],
        samples[i][1, :],
        cmap=plt.get_cmap("Greys"),
        bins=60,
        # range=[[-6, 6], [-6, 6]],
    )
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])

plt.show()


sampler = hmc_tomography.Samplers.RWMH()
posterior = hmc_tomography.Distributions.Himmelblau(temperature=1)
filename = "samples0.h5"
sampler.sample(
    filename,
    posterior,
    proposals=1000000,
    autotuning=True,
    overwrite_existing_file=True,
)
print("\x1b[2K\r")
