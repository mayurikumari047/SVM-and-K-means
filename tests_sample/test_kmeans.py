import unittest
import numpy as np
import clustering, datasets


class TestFFH(unittest.TestCase):
    def setUp(self):
        (self.X, self.Y) = datasets.loadDigits()

    def test_ffh_10(self):

        finalObj = []
        for rep in range(10):
            np.random.seed(1234 + rep)
            mu0 = clustering.initialize_clusters(self.X, 10, 'ffh')
            (mu, z, obj) = clustering.kmeans(self.X, mu0, doPlot=False)
            finalObj.append(obj[-1])

        targetObj = 0.44031610993896342
        self.assertTrue(abs(np.mean(finalObj) - targetObj) <= 1e4)



class TestKMPLUSPLUS(unittest.TestCase):
    def setUp(self):
        (self.X, self.Y) = datasets.loadDigits()

    def test_km_plus_plus_10(self):

        finalObj = []
        for rep in range(20):
            np.random.seed(1234 + rep)
            mu0 = clustering.initialize_clusters(self.X, 10, 'km++')
            (mu, z, obj) = clustering.kmeans(self.X, mu0, doPlot=False)
            finalObj.append(obj[-1])

        targetObj = 0.4392510535744174
        self.assertTrue(abs(np.mean(finalObj) - targetObj) <= 1e4)
    