import h5py as _h5py


class Samples:
    filename: str = None

    def __init__(self, filename):
        self.filename = filename
        try:
            self.file_handle: _h5py.File = _h5py.File(self.filename, "r")
        finally:
            pass

    def __del__(self):
        self.file_handle.close()

    @property
    def raw_samples(self):
        return self.file_handle["samples 0"]
