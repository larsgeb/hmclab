import h5py as _h5py


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
            return self.raw_samples[:, self.burn_in + key][:, None]

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
            return self.raw_samples[:, key]

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
            return self.raw_samples[key1, key2]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file_handle.close()

    @property
    def raw_samples(self):
        return self.file_handle[self.datasetname]
