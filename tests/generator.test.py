import unittest

import numpy as np
import pandas as pd
from src.config import c as c
from src.generator import Generator


class Test_Generator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.df = pd.read_csv(f"{c['WORK_DIR']}/work.csv")
        self.generator = Generator(df=self.df)

    def test_len(self):
        # length in batches
        self.assertEqual(self.generator.__len__(), 580)

    def test_get_one_1(self):
        x, y = self.generator.__getitem__(0)
        self.assertEqual(x[0].dtype, np.uint8)
        self.assertEqual(y[0].dtype, np.float16)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
