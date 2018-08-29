import unittest
import imutil
import os
import numpy as np

EXTENSIONS = ['.jpg', '.png', '.mjpeg', '.mp4']

class TestImutil(unittest.TestCase):

    def setUp(self):
        os.environ['IMUTIL_SHOW'] = '1'
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
        imutil.show(x, save=False)
        imutil.show(x, save=True)
        imutil.show(x, filename='test_output.png', save=False)
        imutil.show(x, filename='test_output.png', save=True)

    def test_video_output(self):
        x = np.zeros((128,128))
        for i in range(10):
            x += 10.
            imutil.show(x, video_filename='test_output.mjpeg')

    def test_display(self):
        x = np.random.uniform(0, 1, size=(128,128))

        # show(display=False) should save a jpg in the cwd
        listing_before = os.listdir('.')
        imutil.show(x, display=False)
        listing_after = os.listdir('.')
        assert len(listing_after) == len(listing_before) + 1

        # show(display=True) should save a jpg, and also call imgcat if available
        imutil.show(x, display=True)


def cleanup():
    files = os.listdir('.')
    for f in files:
        if any(f.endswith(ext) for ext in EXTENSIONS):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
