import os

def get_filename(filepath):

    _, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)

    return filename