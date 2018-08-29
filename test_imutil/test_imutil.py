import unittest
import imutil
import os
import numpy as np

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


def cleanup():
    files = os.listdir('.')
    if 'test_output.jpg' in files:
        os.remove('test_output.jpg')
    if 'test_output.png' in files:
        os.remove('test_output.png')
    if 'test_output.mjpeg' in files:
        os.remove('test_output.mjpeg')


if __name__ == '__main__':
    unittest.main()
