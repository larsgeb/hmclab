import shutil as _shutil

import h5py as _h5py

from hmclab.Helpers.BetterABC import ABCMeta as _ABCMeta
from hmclab.Helpers.AppendNPY import AppendableArray as _AppendableArray
import pickle as _pickle
from time import time as _time
from typing import List as _List, Union as _Union
import numpy as _numpy
import os as _os


class Samples:

    filetype = None
    mode = None
    filename = None
    directory = None

    def __init__(self, filename, burn_in=None, mode="r", overwrite=None):

        filename_stripped, file_extension = _os.path.splitext(filename)

        # Check file extension
        if file_extension == "" or file_extension == ".h5":
            self._inside_context = False
            self.filetype = "HDF5"
            filename = filename_stripped + ".h5"
        elif file_extension == ".npy":
            self.filetype = "NPY"
        else:
            raise AttributeError(
                f"Unkown extension `{file_extension}` for samples file."
            )

        # Check if we are in read or write mode
        self.mode = mode
        self.filename = filename

        if mode == "r":

            # Check for irrelevant parameters in read mode
            if overwrite is not None:
                raise AttributeError("Overwrite is not relevant when writing samples.")

            # Make sure file exists if we try to read.
            if not _os.path.isfile(filename):
                raise FileNotFoundError(
                    f"Trying to read samples file `{filename}` which does not exist."
                )

            # Set up the file to write to
            if self.filetype == "HDF5":
                self.setup_read_hdf5()
            elif self.filetype == "NPY":
                self.setup_read_numpy()
            else:
                raise AttributeError(f"Unkown filetype `{self.filetype}`.")

            # Set the length of the burn-in phase
            if burn_in == None:
                burn_in = 0
            self.burn_in = burn_in

            # Get the last written sample
            self.last_sample = self.read_attribute("write_index")

            # Check if there are any samples left after burn-in
            if self.last_sample <= self.burn_in:
                self.close()
                raise ValueError(
                    f"The burn-in phase is longer than the chain itself. "
                    f"Total samples before burn in: {self.last_sample}"
                )

        elif mode == "w":
            self._buffer = []
            self._buffer_interval = 1
            self._last_append_time = _numpy.nan

            # Check for irrelevant parameters in write mode
            if burn_in is not None:
                raise AttributeError("Burn in is not relevant when writing samples.")

            directory = _os.path.dirname(filename)
            self.directory = directory

            # Check if the directory exists
            if directory != "" and not _os.path.isdir(directory):
                raise NotADirectoryError(
                    f"Trying to write a samples file to a "
                    f"non-existent directory `{directory}`."
                )

            # Set if we are allowed to overwrite samples file
            if overwrite == None:
                overwrite = False
            self.overwrite = overwrite

            # Check if the file does not exist yet
            if (not overwrite) and (
                _os.path.isfile(filename)
                or (self.filetype == "NPY" and _os.path.isfile(f"{filename}.pkl"))
            ):
                if _os.path.isfile(f"{filename}.pkl"):
                    filename += f"` or attributes file `{filename}.pkl"
                raise FileExistsError(
                    f"Trying to write samples to an already existing file `{filename}`."
                )

            # Set up the file to write to
            if self.filetype == "HDF5":
                self.setup_write_hdf5()
            elif self.filetype == "NPY":
                self.setup_write_numpy()
            else:
                raise AttributeError(f"Unkown filetype `{self.filetype}`.")

            # Set the current index of samples to the start of the file
            self.write_attribute("write_index", 0)
            self.write_attribute("last_written_sample", -1)

        else:
            raise AttributeError(f"Unkown file mode `{mode}` for samples file.")

    def setup_write_hdf5(self):

        # Create file, fail if exists and flag == w-
        if self.overwrite:
            flag = "w"
        else:
            flag = "w-"
        self._hdf5_filehandle = _h5py.File(self.filename, flag, libver="latest")

        # Create dataset
        self._hdf5_dataset = self._hdf5_filehandle.create_dataset(
            "samples",
            (0, 0),
            maxshape=(None, None),  # one extra for misfit
            dtype="f8",
            chunks=True,
        )
        # Fill with NaNs
        self._hdf5_dataset.set_fill_value = _numpy.nan

    def setup_write_numpy(self):

        if _os.path.isfile(self.filename) and self.overwrite:
            _os.remove(self.filename)

        self._numpy_appendable_array = _AppendableArray(self.filename)
        self._numpy_attributes = {}

    def setup_read_hdf5(self):
        try:
            self._hdf5_filehandle = _h5py.File(self.filename, "r")
            self._hdf5_dataset = self._hdf5_filehandle["samples"]
        except Exception as e:
            raise ValueError(f"Was not able to open the samples file. Exception: {e}")

    def setup_read_numpy(self):
        self._numpy_array = _numpy.load(self.filename, mmap_mode="r").T
        with open(f"{self.filename}.pkl", "rb") as f:
            self._numpy_attributes = _pickle.load(f)

    def write_attribute(self, name, value):
        assert self.mode == "w"
        if self.filetype == "HDF5":
            self._hdf5_dataset.attrs[name] = value
        elif self.filetype == "NPY":
            self._numpy_attributes[name] = value
            with open(f"{self.filename}.pkl", "wb") as f:
                _pickle.dump(self._numpy_attributes, f)
        else:
            raise AttributeError()

    def read_attribute(self, name):
        if self.filetype == "HDF5":
            return self._hdf5_dataset.attrs[name]
        elif self.filetype == "NPY":
            # if self.mode == "w":
            #     with open(f"{self.filename}.pkl", "rb") as f:
            #         self._numpy_attributes = _pickle.load(f)
            return self._numpy_attributes[name]
        else:
            raise AttributeError()

    def show_all_attributes(self):
        print(self._numpy_attributes)

    def __del__(self):
        self.close()

    def __getitem__(self, key):
        """[ ] operator"""
        if self.filetype == "HDF5":
            return self._hdf5_dataset[:, self.burn_in :][key]
        elif self.filetype == "NPY":
            # TODO
            pass
        else:
            raise AttributeError(f"Unkown filetype `{self.filetype}`.")

    def __enter__(self):
        """Context manager enter, important for HDF5 filehandles"""
        if self.filetype == "HDF5":
            self._inside_context = True
        return self

    def __exit__(self, type, value, traceback):
        """Context manager exit, important for HDF5 filehandles and attributes metadata
        for NumPy"""
        if self.filetype == "HDF5":
            self._inside_context = False

        self.close()

    def close(self):
        if self.mode == "w":
            self.flush_buffer()

        if hasattr(self, "_closed") and self._closed:
            return
        self._closed = True
        if self.filetype == "HDF5":
            # Close HDF5 file if open
            if hasattr(self, "_hdf5_filehandle"):
                self._hdf5_filehandle.close()
        elif self.filetype == "NPY":

            # Close the appendable array
            if hasattr(self, "_numpy_appendable_array"):
                self._numpy_appendable_array.close()

            # Write out the last attributes if open and in write mode
            if hasattr(self, "_numpy_attributes") and self.mode == "w":
                with open(f"{self.filename}.pkl", "wb") as f:
                    _pickle.dump(self._numpy_attributes, f)
        else:
            raise AttributeError(f"Unkown filetype `{self.filetype}`.")

    @property
    def misfits(self):
        if self.filetype == "HDF5":
            return self._hdf5_filehandle["samples"][-1, self.burn_in :][:, None]
        elif self.filetype == "NPY":
            # TODO
            pass
        else:
            raise AttributeError(f"Unkown filetype `{self.filetype}`.")

    @property
    def samples(self):
        return self.numpy[:-1, :]

    @property
    def numpy(self):
        if self.filetype == "HDF5":
            return self._hdf5_filehandle["samples"][:, self.burn_in :]
        elif self.filetype == "NPY":
            return self._numpy_array
        else:
            raise AttributeError(f"Unkown filetype `{self.filetype}`.")

    def append(self, array):
        """RAM Buffered append."""

        self._buffer.append(array.copy())

        # Check if we should write to disk
        if len(self._buffer) > self._buffer_interval:

            delta_time = _time() - self._last_append_time
            self._last_append_time = _time()
            if delta_time < 1.0:
                self._buffer_interval *= 2
            elif delta_time > 10.0:
                self._buffer_interval /= 2
                self._buffer_interval = max(self._buffer_interval, 1)

            self.flush_buffer()

    def flush_buffer(self):
        assert self.mode == "w"
        if len(self._buffer) == 0:
            return

        self._append(_numpy.hstack(self._buffer))
        self._buffer = []

    def _append(self, array):
        """Straight to disk append"""

        assert self.mode == "w"
        if self.filetype == "HDF5":

            new_samples_count = array.shape[1]
            size_before = _numpy.copy(self._hdf5_dataset.shape)
            size_after = size_before.copy()

            # TODO evaluate if this can be avoided
            size_after[0] = array.shape[0]

            size_after[1] += new_samples_count
            self._hdf5_dataset.resize(size_after)
            self._hdf5_dataset[:, (size_before[1] - size_after[1]) :] = array.copy()
            self.write_attribute("write_index", size_after[1])
            self.write_attribute("last_written_sample", size_after[1] - 1)
            self._hdf5_dataset.flush()

        elif self.filetype == "NPY":
            self._numpy_appendable_array.append(_numpy.ascontiguousarray(array.T))

            self.write_attribute(
                "write_index", self.read_attribute("write_index") + array.shape[1]
            )
            self.write_attribute(
                "last_written_sample",
                self.read_attribute("last_written_sample") + array.shape[1],
            )
        else:
            raise AttributeError(f"Unkown filetype `{self.filetype}`.")

    """def print_details(self):
        size = _shutil.get_terminal_size((40, 20))
        width = size[0]
        if _in_notebook():
            width = 80
        print()
        print("{:^{width}}".format("H5 file details", width=width))
        print("━" * width)
        print("{0:30} {1}".format("Filename", self.filename)) 
        dataset = self._hdf5_filehandle["samples"]
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
            print("{0:30} {1}".format(key, details[key]))"""


def combine_samples(
    samples_list: _Union[_List[Samples], _List[str]],
    output_filename=None,
    cull_nan=True,
):
    assert (
        type(samples_list) == list
    ), "Passed sample files/objects are not in list format."

    close_files = False
    ret_obj = None

    if all(isinstance(n, Samples) for n in samples_list):
        pass

    elif all(isinstance(n, str) for n in samples_list):
        close_files = True
        samples_list = [Samples(samples_item) for samples_item in samples_list]

    else:
        raise ValueError(
            "Passed neither only strings to a sample files nor only sample collections."
            " Can't combine samples. "
        )

    # Concatenation is in memory
    if output_filename is None:
        ret_obj = _numpy.hstack([samples_item.numpy for samples_item in samples_list])

        if cull_nan:
            ret_obj = ret_obj[
                :,
                _numpy.logical_not(_numpy.isnan(_numpy.sum(ret_obj, axis=0))),
            ]
    else:
        raise NotImplemented

    if close_files:
        for samples_item in samples_list:
            samples_item.close()

    return ret_obj


def _in_notebook():
    try:
        from IPython import get_ipython

        if (
            not get_ipython() or "IPKernelApp" not in get_ipython().config
        ):  # pragma: no cover
            return False
    except ImportError:
        return False
    return True
