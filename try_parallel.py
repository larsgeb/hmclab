import hmclab
import matplotlib.pyplot as plt

resample = True

# Set up parallel Markov chains by generating 3 copies of everything -------------------
chains = 3

# 3 samplers
samplers = [hmclab.Samplers.RWMH() for i in range(chains)]

# 3 posteriors (that are not the same!)
posteriors = [
    hmclab.Distributions.Himmelblau(temperature=temp) for temp in [100, 10, 3]
]

# 3 separate sample files
filenames = [f"samples_parallel_{i}.h5" for i in range(chains)]

if resample:
    # Execute parallel sampling
    hmclab.Samplers.ParallelSampleSMP(
        samplers, filenames, posteriors, 100000, exchange_interval=100
    )


samples_objs = [hmclab.Samples(filename) for filename in filenames]

samples_RWMH_parallel = [so.numpy for so in samples_objs]

for so in samples_objs:
    so.close()
    del so

print("\x1b[2K\r")

sampler = hmclab.Samplers.RWMH()
posterior = posteriors[-1]
filename_RWMH = "samples0.h5"

if resample:
    sampler.sample(
        filename_RWMH,
        posterior,
        proposals=100000,
        autotuning=True,
        overwrite_existing_file=True,
    )
samples_obj_RWMH = hmclab.Samples(filename_RWMH)
samples_RWMH = samples_obj_RWMH.numpy
print("\x1b[2K\r")


plt.figure(figsize=(8, 4))
for i in range(chains):
    plt.subplot(1, chains + 1, int(i + 1), aspect="equal")
    plt.hist2d(
        samples_RWMH_parallel[i][0, :],
        samples_RWMH_parallel[i][1, :],
        cmap=plt.get_cmap("Greys"),
        bins=60,
        # range=[[-6, 6], [-6, 6]],
    )
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])

plt.subplot(1, chains + 1, chains + 1, aspect="equal")
plt.hist2d(
    samples_RWMH[0, :],
    samples_RWMH[1, :],
    cmap=plt.get_cmap("Greys"),
    bins=60,
    # range=[[-6, 6], [-6, 6]],
)
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.show()
