import random
import numpy as np


class DataFormatter:

    @staticmethod
    def format_data(tender, test_size=0.1):
        random.shuffle(tender)

        tender = np.array(tender)

        testing_size = int(test_size*len(tender))
        train_x = list(tender[:, 0][: -testing_size])
        train_y = list(tender[:, 1][: -testing_size])

        test_x = list(tender[:, 0][-testing_size:])
        test_y = list(tender[:, 1][-testing_size:])

        return train_x, train_y, test_x, test_y
