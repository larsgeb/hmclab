import h5py as _h5py
import os as _os
import shutil as _shutil


class Samples:
    """A class to handle generated samples files more easily.

    """

    filename: str = None

    datasetname = "samples_0"

    def __init__(self, filename, burn_in: int = 0):
        self.filename = filename
        try:
            self.file_handle: _h5py.File = _h5py.File(self.filename, "r")
        except Exception as e:
            raise ValueError(f"Was not able to open the samples file. Exception: {e}")
        self.burn_in = burn_in

        # Property indicating that sampling is terminated prematurely
        self.last_sample = self.file_handle[self.datasetname].attrs["end_of_samples"]

        if self.last_sample <= self.burn_in:
            raise ValueError("The burn-in phase is longer than the chain itself.")

    def __del__(self):
        self.file_handle.close()

    def __getitem__(self, key):
        """This operator overloads the [] brackets to correct for burn in.
        
        The operator overload takes care of the burn-in phase sample discard."""

        if type(key) == int:
            # Check if the indexed sample is actual drawn.
            if key > self.last_sample:
                raise ValueError("Index out of range")
            # Access a single samples. The none keyword forces a column vector.
            return self.raw_samples_hdf[:, self.burn_in + key][:, None]

        elif type(key) == slice:

            start = None
            stop = None
            step = key.step

            if key.start is not None:
                start = key.start + self.burn_in
            else:
                start = self.burn_in
            if key.stop is not None:
                stop = key.stop + self.burn_in

            # Correct for the possible sampling termination
            if stop is None:
                stop = self.last_sample

            if start > stop:
                raise ValueError("Index out of range")

            key = slice(start, stop, step)

            print(key, self.last_sample)

            # Access multiple samples
            return self.raw_samples_hdf[:, key]

        elif type(key) == tuple:

            key1 = key[0]
            key2 = key[1]

            if type(key2) == int:
                key2 = slice(key2 + self.burn_in, key2 + self.burn_in + 1, None)
            else:
                start = None
                stop = None
                step = key2.step

                if key2.start is not None:
                    start = key2.start + self.burn_in
                else:
                    start = self.burn_in
                if key2.stop is not None:
                    stop = key2.stop + self.burn_in

                # Correct for the possible sampling termination
                if stop is None:
                    stop = self.last_sample

                if start > stop:
                    raise ValueError("Index out of range")

                key2 = slice(start, stop, step)

            # Access multiple samples
            return self.raw_samples_hdf[key1, key2]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file_handle.close()

    def close(self):
        self.file_handle.close()

    @property
    def misfits(self):
        return self.file_handle[self.datasetname][-1, :][:, None]

    @property
    def raw_samples(self):
        return self.file_handle[self.datasetname][:-1, :].T

    @property
    def raw_samples_hdf(self):
        return self.file_handle[self.datasetname]

    def print_details(self):

        size = _shutil.get_terminal_size((80, 20))  # pass fallback

        print()
        print("{:^{width}}".format("H5 file details", width=size[0]))
        print("━" * size[0])
        print("{0:30} {1}".format("Filename", self.filename))
        print("{0:30} {1}".format("Dataset", self.datasetname))

        dataset = self.file_handle[self.datasetname]
        details = dict(
            (key, value) for key, value in _h5py.AttributeManager(dataset).items()
        )

        # Print common attributes
        print()
        print("{:^{width}}".format("Sampling attributes", width=size[0]))
        print("━" * size[0])
        print("{0:30} {1}".format("Sampler", details["sampler"]))
        print("{0:30} {1}".format("Requested proposals", details["proposals"]))
        print("{0:30} {1}".format("Proposals saved to disk", details["end_of_samples"]))
        print("{0:30} {1}".format("Acceptance rate", details["acceptance_rate"]))
        print("{0:30} {1}".format("Online thinning", details["online_thinning"]))
        print("{0:30} {1}".format("Sampling start time", details["start_time"]))

        details.pop("sampler")
        details.pop("proposals")
        details.pop("end_of_samples")
        details.pop("acceptance_rate")
        details.pop("online_thinning")
        details.pop("start_time")

        print()
        print("{:^{width}}".format("Sampler specific attributes", width=size[0]))
        print("━" * size[0])
        for key in details:
            print("{0:30} {1}".format(key, details[key]))
