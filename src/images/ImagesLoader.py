from skimage import io
import numpy as np


class ImagesLoader:

    def __init__(self):
        self.tender = []
        self.classes = []

    def load_person_data(self, filesPath):
        person = io.ImageCollection(filesPath)
        return person

    def add_new_class_to_tensor(self, data_vector):
        self.classes.append(1)

        for index in range(0, len(self.tender)):
            self.tender[index][1].append(0)

        for data in data_vector:
            processed_data = np.array(data).flatten()  ## flattens arrays to one array [R G B A  R G B A] or [L L L L]
            self.tender.append([processed_data, self.classes[:]])

        self.classes[-1] = 0

    def print_current_tender(self):
        for data in self.tender:
            print(data[:][1])

        print(len(self.tender))

    def load_dara_and_add_as_class(self, filesPath):
        person01 = self.load_person_data(filesPath)
        self.add_new_class_to_tensor(person01)

    def get_data(self):
        return self.tender, len(self.classes)
