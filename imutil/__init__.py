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


# An included default monospace font
def get_font_file():
    return os.path.join(os.path.dirname(__file__), 'DejaVuSansMono.ttf')


# Input: Numpy array containing one or more images
# Output: JPG encoded image bytes (or an alternative format if specified)
def encode_jpg(pixels, resize_to=None, img_format='JPEG'):
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)
    # Convert to RGB to avoid "Cannot handle this data type"
    if pixels.shape[-1] < 3:
        pixels = np.repeat(pixels, 3, axis=-1)
    img = Image.fromarray(pixels.astype(np.uint8))
    if resize_to:
        img = img.resize(resize_to)
    fp = BytesIO()
    img.save(fp, format=img_format)
    return fp.getvalue()


# Input: Filename, or JPG bytes
# Output: Numpy array containing images
def decode_jpg(jpg, crop_to_box=None, resize_to=(224,224), pil=False):
    if jpg.startswith('\xFF\xD8'):
        # Input is a JPG buffer
        img = Image.open(BytesIO(jpg))
    else:
        # Input is a filename
        img = Image.open(jpg)

    img = img.convert('RGB')
    if crop_to_box:
        # Crop to bounding box
        x0, x1, y0, y1 = crop_to_box
        width, height = img.size
        absolute_box = (x0 * width, y0 * height, x1 * width, y1 * height)
        img = img.crop((int(i) for i in absolute_box))
    if resize_to:
        img = img.resize(resize_to)
    if pil:
        return img
    return np.array(img).astype(float)


figure = []
def add_to_figure(data):
    figure.append(data)


def show_figure(**kwargs):
    global figure
    show(np.array(figure), **kwargs)
    figure = []


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
    if len(pixels.shape) == 4:
        pixels = pixels.transpose((0,2,3,1))
    elif len(pixels.shape) == 3 and pixels.shape[0] in (1, 3):
        pixels = pixels.transpose((1,2,0))
    return pixels


# Swiss-army knife for putting an image on the screen
# Accepts numpy arrays, PIL Image objects, or jpgs
# Numpy arrays can consist of multiple images, which will be collated
def show(
        data,
        verbose=False,
        display=True,
        save=True,
        filename=None,
        box=None,
        video_filename=None,
        resize_to=None,
        normalize_color=True,
        caption=None,
        font_size=16,
        return_pixels=False):
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
        pixels = decode_jpg(data, resize_to=resize_to)
    else:
        print('imutil.show() handling unknown type {}'.format(type(data)))
        pixels = np.array(data)

    # Split non-RGB images into sets of monochrome images
    if pixels.shape[-1] not in (1, 3):
        pixels = np.expand_dims(pixels, axis=-1)

    # Expand matrices to monochrome images
    while len(pixels.shape) < 3:
        pixels = np.expand_dims(pixels, axis=-1)

    # Reduce lists of images to a single image
    while len(pixels.shape) > 3:
        pixels = combine_images(pixels)

    # Normalize pixel intensities
    if normalize_color and pixels.max() > pixels.min():
        pixels = (pixels - pixels.min()) * 255. / (pixels.max() - pixels.min())

    # Resize image to desired shape
    if resize_to:
        if pixels.shape[-1] == 1:
            pixels = pixels.repeat(3, axis=-1)
        img = Image.fromarray(pixels.astype('uint8'))
        img = img.resize(resize_to)
        pixels = np.array(img)

    # Draw a bounding box onto the image
    if box is not None:
        draw_box(pixels, box)

    # Draw text into the image
    if caption is not None:
        pixels = pixels.squeeze()
        img = Image.fromarray(pixels.astype('uint8'))
        font = ImageFont.truetype(get_font_file(), font_size)
        draw = ImageDraw.Draw(img)
        textsize = draw.textsize(caption, font=font)
        # TODO: issues with fill
        #draw.rectangle([(0, 0), textsize], fill=(0,0,0,128))
        draw.rectangle([(0, 0), textsize])
        #draw.multiline_text((0,0), caption, font=font, fill=(255,255,255))
        draw.multiline_text((0,0), caption, font=font)
        pixels = np.array(img)

    # Set a default filename if one does not exist
    if save and filename is None and video_filename is None:
        output_filename = '{}.jpg'.format(int(time.time() * 1000))
    elif filename is None:
        output_filename = tempfile.NamedTemporaryFile(suffix='.jpg').name

    # Write the file itself
    ensure_directory_exists(output_filename)
    with open(output_filename, 'wb') as fp:
        save_format = 'PNG' if output_filename.endswith('.png') else 'JPEG'
        fp.write(encode_jpg(pixels, img_format=save_format))
        fp.flush()

    should_show = os.environ.get('IMUTIL_SHOW') and len(os.environ['IMUTIL_SHOW']) > 0 and spawn.find_executable('imgcat')
    if display and should_show:
        print('\n' * 4)
        print('\033[4F')
        subprocess.check_call(['imgcat', output_filename])
        print('\033[4B')
    elif verbose:
        print("Saved image size {} as {}".format(pixels.shape, output_filename))

    # Output JPG files can be collected into a video with ffmpeg -i *.jpg
    if video_filename:
        ensure_directory_exists(video_filename)
        open(video_filename, 'ab').write(encode_jpg(pixels))

    if filename is None:
        os.remove(output_filename)

    if return_pixels:
        return pixels


def ensure_directory_exists(filename):
    # Assume whatever comes after the last / is the filename
    tokens = filename.split('/')[:-1]
    # Perform a mkdir -p on the rest of the path
    path = '/'.join(tokens)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def encode_video(video_filename, loopy=False):
    output_filename = video_filename.replace('mjpeg', 'mp4')
    print('Encoding video {}'.format(video_filename))
    # TODO: Tokenize, use subprocess, validate filenames and "&& rm output" in Python
    cmd = 'ffmpeg -hide_banner -nostdin -loglevel panic -y -i {0} '.format(video_filename)
    if loopy:
        cmd += '-filter_complex "[0]reverse[r];[0][r]concat" '
    cmd += '{} && rm {}'.format(output_filename, video_filename)
    # TODO: security lol
    os.system(cmd)


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
def combine_images(images, stack_width=None):
    num_images = images.shape[0]
    input_height = images.shape[1]
    input_width = images.shape[2]
    optional_dimensions = images.shape[3:]

    if not stack_width:
        stack_width = int(math.sqrt(num_images))
    stack_height = int(math.ceil(float(num_images) / stack_width))

    output_width = stack_width * input_width
    output_height = stack_height * input_height

    output_shape = (output_height, output_width) + optional_dimensions
    image = np.zeros(output_shape, dtype=images.dtype)

    for idx in range(num_images):
        i = int(idx / stack_width)
        j = idx % stack_width
        a0, a1 = i * input_height, (i+1) * input_height
        b0, b1 = j * input_width, (j+1) * input_width
        image[a0:a1, b0:b1] = images[idx]
    return image


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
