import bmp
import unittest


class MNISTTest(unittest.TestCase):
    def test_load_bmp(self):
        image = bmp.load('test_image.bmp')
        self.assertIsNotNone(image)


if __name__ == '__main__':
    unittest.main()
