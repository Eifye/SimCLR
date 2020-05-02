import os

def list_images(dir, ext_filter=['.jpg', '.png', '.tiff', '.bmp']):

    files = os.listdir(dir)
    dst = []

    for fn in files:
        ext = os.path.splitext(fn)[1]
        ext = ext.lower()

        if not ext in ext_filter:
            continue

        dst.append(os.path.join(dir, fn))

    return dst

def list_image_dirs(dirs, ext_filter=['.jpg', '.png', '.tiff', '.bmp']):

    dst = []
    for dir in dirs:
        files = list_images(dir, ext_filter)
        dst.extend(files)

    return dst