from PIL import Image, ImageOps

def load_image_fix_exif(path):
    """
    Load an image and fix EXIF orientation. Return a PIL.Image in RGB.
    """
    img = Image.open(path)
    # This call rotates image to upright according to EXIF, if present.
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
