import unittest
import imutil
import os
import numpy as np

EXTENSIONS = ['.jpg', '.png', '.mjpeg', '.mp4']

class TestImutil(unittest.TestCase):

    def setUp(self):
        os.environ['IMUTIL_SHOW'] = ''
        cleanup()

    def tearDown(self):
        cleanup()

    def test_default_filename(self):
        x = np.random.uniform(0, 1, size=(128,128))
        imutil.show(x)

    def test_jpg_output(self):
        x = np.random.uniform(0, 1, size=(128,128))
        imutil.show(x, filename='test_output.jpg')

    def test_png_output(self):
        x = np.random.uniform(0, 1, size=(128,128))
        imutil.show(x, filename='test_output.png')

    def test_save(self):
        x = np.random.uniform(0, 1, size=(128,128))

        # Images should be saved with a default name if save=True
        listing_before = os.listdir('.')
        imutil.show(x, save=False)
        imutil.show(x, save=True)
        listing_after = os.listdir('.')
        assert len(listing_after) == len(listing_before) + 1

        # Even if filename is specified, if save=False then do not save
        listing_before = os.listdir('.')
        imutil.show(x, filename='test_output.png', save=False)
        imutil.show(x, filename='test_output.png', save=True)
        listing_after = os.listdir('.')
        assert len(listing_after) == len(listing_before) + 1

    def test_video_output(self):
        # If video_filename is specified, then save ONLY to video
        x = np.zeros((128,128))
        listing_before = os.listdir('.')
        for i in range(10):
            x += 10.
            imutil.show(x, video_filename='test_output.mjpeg')
        listing_after = os.listdir('.')
        assert len(listing_after) == len(listing_before) + 1

    def test_display(self):
        x = np.random.uniform(0, 1, size=(128,128))

        # show(display=False) should save a jpg in the cwd
        imutil.show(x, display=False)

        # show(display=True) should do the same thing
        imutil.show(x, display=True)

        # show(display=True) with IMUTIL_SHOW set should run imgcat if available
        os.environ['IMUTIL_SHOW'] = '1'
        imutil.show(x, display=True, caption='HELLO PIXELS')
        # assert imgcat was called
        os.environ['IMUTIL_SHOW'] = ''

    def test_resize(self):
        x = np.random.uniform(0, 1, size=(128,128))
        x_resized = imutil.show(x, resize_height=512, resize_width=150, return_pixels=True)
        assert x_resized.shape == (512, 150, 3)

    def test_reshape_ndarray(self):
        # A HWC monochrome image tensor should convert into an RGB image
        x = np.random.uniform(0, 1, size=(100, 100, 1))
        assert imutil.reshape_ndarray_into_rgb(x).shape == (100, 100, 3)

        # A HWC RGB image tensor should convert into an RGB image
        x = np.random.uniform(0, 1, size=(100, 100, 3))
        assert imutil.reshape_ndarray_into_rgb(x).shape == (100, 100, 3)

        # A 2D array should convert into an RGB image
        x = np.random.uniform(0, 1, size=(128,128))
        assert imutil.reshape_ndarray_into_rgb(x).shape == (128, 128, 3)

        # A 1D vector should convert into a 1-row RGB image
        x = np.random.uniform(0, 1, size=(1000))
        assert imutil.reshape_ndarray_into_rgb(x).shape == (1, 1000, 3)

        # One hundred 16x16 icons (a BHWC tensor) should produce an RGB image
        x = np.random.uniform(0, 1, size=(100, 16, 16))
        assert imutil.reshape_ndarray_into_rgb(x).shape == (160, 160, 3)

        # Twenty five sets of one hundred 16x16 icons
        x = np.random.uniform(0, 1, size=(25, 100, 16, 16))
        assert imutil.reshape_ndarray_into_rgb(x).shape[-1] == 3

    def test_reshape_normalize(self):
        x = np.random.normal(size=(128,128))
        # Reshaping the image should rescale it to 0,255 by default
        reshaped = imutil.show(x, resize_height=480, resize_width=640, return_pixels=True)
        assert reshaped.min() >= 0

        # Reshaping with normalize=False should leave the scale unchanged
        reshaped_denorm = imutil.show(x, resize_height=480, resize_width=640, normalize=False, return_pixels=True)
        assert reshaped_denorm.min() < 0


def cleanup():
    files = os.listdir('.')
    for f in files:
        if any(f.endswith(ext) for ext in EXTENSIONS):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
