from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors
def open_file_read_binary(uri):
    try:
        return file_io.FileIO(uri, mode='rb')
    except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')
    
def open_file_write_binary(uri):
    return file_io.FileIO(uri, mode='w')
