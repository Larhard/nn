import unittest
import MNIST


class MNISTTest(unittest.TestCase):
    def test_get_empty_data(self):
        images, labels = MNIST.get_data({})
        self.assertEqual(len(images), 0)
        self.assertEqual(len(labels), 0)

    def test_get_some_data(self):
        requested = {1, 3}
        images, labels = MNIST.get_data(requested)
        self.assertNotEqual(len(images), 0)
        self.assertEqual(len(images), len(labels))
        self.assertEqual(set(labels), set(requested))

    def test_test_data(self):
        requested = {1, 8, 9, 0}
        images, labels = MNIST.get_data(requested, dataset='test')
        self.assertNotEqual(len(images), 0)
        self.assertEqual(len(images), len(labels))
        self.assertEqual(set(labels), set(requested))

if __name__ == '__main__':
    unittest.main()