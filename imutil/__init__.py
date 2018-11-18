import math
import os
import tempfile
import time
import subprocess
import pathlib
from distutils import spawn
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO


# A general-purpose function for turning data into an image on the screen
# If imgcat is installed and IMUTIL_SHOW=1 then the image will be
# displayed inline in the terminal
def show(
        data,
        verbose=False,
        display=True,
        save=True,
        filename=None,
        box=None,
        video_filename=None,
        resize_to=None,
        resize_height=None,
        resize_width=None,
        normalize=True,
        caption=None,
        font_size=16,
        stack_width=None,
        img_padding=0,
        return_pixels=False):
    # Handle special parameter combinations
    if video_filename:
        save = False
    if resize_to:
        resize_width, resize_height = resize_to

    # Convert ANY input into a np.array
    pixels = load(data, verbose=verbose)
    assert type(pixels) == np.ndarray

    # Convert ANY np.array to shape (height, width, 3)
    pixels = reshape_ndarray_into_rgb(pixels, stack_width, img_padding)
    assert len(pixels.shape) == 3

    # Normalize pixel intensities
    if normalize:
        pixels, min_val, max_val = normalize_color(pixels)

    # Resize image to desired shape
    if resize_height or resize_width:
        pixels = resize(pixels, resize_height, resize_width)

    # Draw a bounding box onto the image
    if box is not None:
        draw_box(pixels, box)

    # Draw a text caption above the image
    if caption is not None:
        pixels = draw_text_caption(pixels, caption, font_size)

    # Set a default filename if one does not exist
    if filename is None:
        filename = '{}.jpg'.format(int(time.time() * 1000))

    # Write the file itself
    ensure_directory_exists(filename)
    with open(filename, 'wb') as fp:
        save_format = 'PNG' if filename.endswith('.png') else 'JPEG'
        fp.write(encode_image(pixels, img_format=save_format))
        fp.flush()

    if display:
        display_image_on_screen(filename)

    # The MJPEG format is a concatenation of JPEG files, and can be converted
    # into another format with eg. ffmpeg -i frames.mjpeg output.mp4
    if video_filename:
        ensure_directory_exists(video_filename)
        with open(video_filename, 'ab') as fp:
            fp.write(encode_image(pixels, img_format='JPEG'))

    if not save:
        os.remove(filename)

    if return_pixels:
        return pixels


# A general-purpose image loading function
# Accepts numpy arrays, PIL Image objects, or jpgs
# Numpy arrays can consist of multiple images, which will be collated
def load(data, resize_to=None, crop_to_box=None, verbose=False):
    # Munge data to allow input filenames, pixels, PIL images, etc
    if type(data) == type(np.array([])):
        pixels = data
    elif type(data) == Image.Image:
        pixels = np.array(data)
    elif type(data).__name__ in ['FloatTensor', 'Tensor', 'Variable']:
        pixels = convert_pytorch_tensor_to_pixels(data)
    elif hasattr(data, 'savefig'):
        pixels = convert_fig_to_pixels(data)
    elif type(data).__name__ == 'AxesSubplot':
        pixels = convert_fig_to_pixels(data.get_figure())
    elif hasattr(data, 'startswith'):
        pixels = decode_image_from_string(data)
    else:
        if verbose:
            print('imutil.load() handling unknown type {}'.format(type(data)))
        pixels = np.array(data)
    # Resize image to desired shape
    if resize_to:
        height, width = resize_to
        pixels = resize(pixels, height, width)
    if crop_to_box:
        raise NotImplementedError('imutil.load() crop_to_box')
    return pixels


def convert_fig_to_pixels(matplot_fig):
    # Hack: Write entire figure to file, then re-load it
    # Could be done faster in memory
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        matplot_fig.savefig(tmp.name)
        pixels = np.array(Image.open(tmp.name))
    # Discard the alpha channel, use RGB
    return pixels[:,:,:3]


def convert_pytorch_tensor_to_pixels(data):
    if data.requires_grad:
        data = data.detach()
    pixels = data.cpu().numpy()
    # Special case: Assume Pytorch tensors will be BCHW, convert to BHWC
    if len(pixels.shape) == 4:
        pixels = pixels.transpose((0,2,3,1))
    elif len(pixels.shape) == 3 and pixels.shape[0] in (1, 3):
        pixels = pixels.transpose((1,2,0))
    return pixels


# Input: Filename, or JPG bytes
# Output: Numpy array containing images
def decode_image_from_string(data):
    if data.startswith('\xFF\xD8'):
        # Input is a JPG buffer
        img = Image.open(BytesIO(data))
    else:
        # Input is a filename
        img = Image.open(os.path.expanduser(data))

    img = img.convert('RGB')
    return np.array(img).astype(float)


# pixels: np.array of ANY nonzero dimensionality
# Output: np.array of shape (height, width, 3)
def reshape_ndarray_into_rgb(pixels, stack_width=None, img_padding=0):
    # Special cases: low-dimensional inputs
    if len(pixels.shape) == 1:
        # One-dimensional input: convert to (1 x width x 1)
        pixels = np.expand_dims(pixels, axis=0)
        pixels = np.expand_dims(pixels, axis=-1)
    elif len(pixels.shape) == 2:
        # Two-dimensional input: convert to (height x width x 1)
        pixels = np.expand_dims(pixels, axis=-1)

    n_channels = pixels.shape[-1]
    if n_channels == 1:
        # Convert monochrome to RGB by broadcasting luma
        pixels = np.repeat(pixels, repeats=3, axis=-1)
    elif n_channels != 3:
        # Convert anything else to RGB by making each channel a separate image
        pixels = np.expand_dims(pixels, axis=-1)
        pixels = np.repeat(pixels, repeats=3, axis=-1)

    # Combine lists of images into a single tiled image
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels, stack_width=stack_width, img_padding=img_padding)
    return pixels


# Input: A sequence of images, where images[0] is the first image
# Output: A single image, containing the input images tiled together
# Each input image can be 2-dim monochrome, 3-dim rgb, or more
# Examples:
# Input (4 x 256 x 256 x 3) outputs (512 x 512 x 3)
# Input (4 x 256 x 256) outputs (512 x 512)
# Input (3 x 256 x 256 x 3) outputs (512 x 512 x 3)
# Input (100 x 64 x 64) outputs (640 x 640)
# Input (99 x 64 x 64) outputs (640 x 640) (with one blank space)
# Input (100 x 64 x 64 x 17) outputs (640 x 640 x 17)
def combine_images(images, stack_width=None, img_padding=0):
    num_images = images.shape[0]
    input_height = images.shape[1]
    input_width = images.shape[2]
    optional_dimensions = images.shape[3:]

    if not stack_width:
        stack_width = int(math.sqrt(num_images))
    stack_height = int(math.ceil(float(num_images) / stack_width))

    output_width = stack_width * (input_width + img_padding)
    output_height = stack_height * (input_height + img_padding)

    output_shape = (output_height, output_width) + optional_dimensions
    image = np.ones(output_shape, dtype=images.dtype)

    for idx in range(num_images):
        i = int(idx / stack_width)
        j = idx % stack_width
        a0 = i * (input_height + img_padding) + img_padding//2
        b0 = j * (input_width + img_padding) + img_padding//2
        image[a0:a0 + input_height, b0:b0 + input_width] = images[idx]
    return image


# pixels: np.array of shape (height, width, 3)
def resize(pixels, resize_height, resize_width):
    current_height, current_width, channels = pixels.shape
    if resize_height is None:
        resize_height = current_height
    if resize_width is None:
        resize_width = current_width
    from skimage.transform import resize
    # Normalize to the acceptable range -1, 1
    maxval = pixels.max()
    return maxval * resize(pixels / maxval, (resize_height, resize_width), mode='reflect', anti_aliasing=True)


def normalize_color(pixels, normalize_to=255.):
    min_val, max_val = pixels.min(), pixels.max()
    if min_val == max_val:
        return pixels, min_val, max_val
    pixels = (pixels - min_val) / (max_val - min_val)
    pixels *= normalize_to
    return pixels, min_val, max_val


# pixels: np.array of shape (height, width, 3)
# caption: string, may contain newlines
# font_size: integer
# output: np.array of shape (height + caption_height, width, 3)
def draw_text_caption(pixels, caption, font_size=12):
    height, width, channels = pixels.shape
    font = ImageFont.truetype(get_font_file(), font_size)
    _, caption_height = font.getsize(caption)

    # Scale height to the nearest multiple of 4 for libx264 et al
    new_height = ((height + caption_height + 1) // 4) * 4

    new_pixels = np.zeros((new_height, width, channels), dtype=np.uint8)
    new_pixels[-height:] = pixels
    img = Image.fromarray(new_pixels)

    draw = ImageDraw.Draw(img)
    textsize = draw.textsize(caption, font=font)
    draw.rectangle([(0, 0), textsize], fill=(0,0,0,128))
    draw.multiline_text((0,0), caption, font=font)
    pixels = np.array(img)
    return pixels


# An included default monospace font
def get_font_file():
    return os.path.join(os.path.dirname(__file__), 'DejaVuSansMono.ttf')


# Input: Numpy array containing one or more images
# Output: JPG encoded image bytes (or an alternative format if specified)
def encode_image(pixels, img_format='JPEG'):
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)
    # Convert to RGB to avoid "Cannot handle this data type"
    if pixels.shape[-1] < 3:
        pixels = np.repeat(pixels, 3, axis=-1)
    img = Image.fromarray(pixels.astype(np.uint8))
    fp = BytesIO()
    img.save(fp, format=img_format)
    return fp.getvalue()


figure = []
def add_to_figure(data):
    figure.append(data)


def show_figure(**kwargs):
    global figure
    show(np.array(figure), **kwargs)
    figure = []


def ensure_directory_exists(filename):
    # Assume whatever comes after the last / is the filename
    tokens = filename.split('/')[:-1]
    # Perform a mkdir -p on the rest of the path
    path = '/'.join(tokens)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def display_image_on_screen(filename):
    should_show = os.environ.get('IMUTIL_SHOW') and len(os.environ['IMUTIL_SHOW']) > 0 and spawn.find_executable('imgcat')
    if not should_show:
        return
    print('\n' * 4)
    print('\033[4F')
    subprocess.check_call(['imgcat', filename])
    print('\033[4B')


def encode_video(video_filename, loopy=False):
    output_filename = video_filename.replace('mjpeg', 'mp4')
    print('Encoding MJPEG video {}'.format(video_filename))
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', '-panic', '-y', '-i', video_filename]
    if loopy:
        cmd += ['-filter_complex', '"[0]reverse[r];[0][r]concat"']
    cmd += [output_filename]
    subprocess.run(cmd)

    if os.path.exists(output_filename):
        print('Finished encoding video {}'.format(output_filename))
        os.remove(video_filename)
    else:
        print('imutil warning: Failed to create video file {}, is ffmpeg installed?'.format(output_filename))


def draw_box(img, box, color=1.0):
    height, width, channels = img.shape
    if all(0 < i < 1.0 for i in box):
        box = np.multiply(box, (width, width, height, height))
    x0, x1, y0, y1 = (int(val) for val in box)
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    img[y0:y1,x0] = color
    img[y0:y1,x1] = color
    img[y0,x0:x1] = color
    img[y1,x0:x1] = color


class VideoMaker():
    loopy = False

    def __init__(self, filename):
        self.filename = filename
        self.finished = False
        if self.filename.endswith('.mp4'):
            self.filename = self.filename[:-4]
        if not self.filename.endswith('mjpeg'):
            self.filename = self.filename + '.mjpeg'

    def write_frame(self,
                    frame,
                    font_size=12,
                    **kwargs):
        if self.finished:
            raise ValueError("Video is finished, cannot write frame")
        show(frame,
            video_filename=self.filename,
            font_size=font_size,
            display=False,
            **kwargs)

    def finish(self):
        if not self.finished:
            encode_video(self.filename, loopy=self.loopy)
        self.finished = True

    def __del__(self):
        if not self.finished:
            self.finish()


class VideoLoop(VideoMaker):
    loopy = True


# An easy but inefficient way to draw text onto an image
def text(pixels, text, x=0, y=0, font_size=12, color=(0,0,0,255)):
    from PIL import Image, ImageFont, ImageDraw
    pixels = show(pixels, display=False, save=False, return_pixels=True)
    pixels = (pixels * 255).astype(np.uint8)
    # Hack: convert monochrome to RGB
    if pixels.shape[-1] == 1:
        pixels = np.stack([pixels.squeeze(), pixels.squeeze(), pixels.squeeze()], axis=-1)
    img = Image.fromarray(pixels)
    font_file = get_font_file()
    font = ImageFont.truetype(font_file, font_size)
    draw = ImageDraw.Draw(img)
    textsize = draw.textsize(text, font=font)
    draw.multiline_text((x,y), text, font=font, fill=color)
    pixels = np.array(img)
    return pixels
