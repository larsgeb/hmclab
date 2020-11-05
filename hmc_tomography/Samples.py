import shutil as _shutil

import h5py as _h5py


class Samples:
    """A class to handle generated samples files.

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
        self.last_sample = self.file_handle[self.datasetname].attrs["write_index"]

        if self.last_sample <= self.burn_in:
            raise ValueError("The burn-in phase is longer than the chain itself.")

    def __del__(self):
        self.file_handle.close()

    def __getitem__(self, key):
        """This operator overloads the [] brackets to correct for burn in.

        The operator overload takes care of the burn-in phase sample discard."""
        return self.file_handle[self.datasetname][:, self.burn_in : -1][key]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file_handle.close()

    def close(self):
        self.file_handle.close()

    @property
    def misfits(self):
        return self.file_handle[self.datasetname][-1, self.burn_in : -1][:, None]

    @property
    def numpy(self):
        return self.file_handle[self.datasetname][:, self.burn_in : -1]

    @property
    def h5(self):
        return self.file_handle[self.datasetname]

    def print_details(self):

        size = _shutil.get_terminal_size((40, 20))
        width = size[0]
        if _in_notebook():
            width = 80

        print()
        print("{:^{width}}".format("H5 file details", width=width))
        print("━" * width)
        print("{0:30} {1}".format("Filename", self.filename))
        print("{0:30} {1}".format("Dataset", self.datasetname))

        dataset = self.file_handle[self.datasetname]
        details = dict(
            (key, value) for key, value in _h5py.AttributeManager(dataset).items()
        )

        # Print common attributes
        print()
        print("{:^{width}}".format("Sampling attributes", width=width))
        print("━" * width)
        print("{0:30} {1}".format("Sampler", details["sampler"]))
        print("{0:30} {1}".format("Requested proposals", details["proposals"]))
        print("{0:30} {1}".format("Online thinning", details["online_thinning"]))
        print(
            "{0:30} {1:.2f}".format(
                "Proposals per second",
                details["online_thinning"]
                * details["write_index"]
                / details["runtime_seconds"],
            )
        )
        print("{0:30} {1}".format("Proposals saved to disk", details["write_index"]))
        print("{0:30} {1:.2f}".format("Acceptance rate", details["acceptance_rate"]))
        print("{0:30} {1}".format("Sampler initiate time", details["start_time"]))
        print("{0:30} {1}".format("Sampler terminate time", details["end_time"]))

        details.pop("sampler")
        details.pop("proposals")
        details.pop("write_index")
        details.pop("acceptance_rate")
        details.pop("online_thinning")
        details.pop("start_time")
        details.pop("end_time")
        details.pop("last_written_sample")
        details.pop("runtime_seconds")
        details.pop("runtime")

        print()
        print("{:^{width}}".format("Sampler specific attributes", width=width))
        print("━" * width)
        for key in details:
            print("{0:30} {1}".format(key, details[key]))


def _in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True
