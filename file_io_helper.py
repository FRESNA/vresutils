import os, shutil

def ensure_mkdir(path):
    '''Conflict free mkdir.'''
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def copy_without_overwrite(src, dest):
    # Open the file and dont do anything if it exists
    try:    
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        if os.path.isfile(dest):
            return
        else: raise

    # Copy the file and automatically close files at the end
    with os.fdopen(fd,'w') as f:
        with open(src) as sf:
            shutil.copyfileobj(sf, f)

