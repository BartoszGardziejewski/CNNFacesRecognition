from src.data.DataFormatter import DataFormatter
from src.images.ImagesLoader import ImagesLoader
from src.neuralnetwork.NeuralNetworkImages import NeuralNetwork as ImageNeuralNetwork
from src.neuralnetwork.NeuralNetworkMinistData import NeuralNetwork as MinistNeuralNetwork


def run_minist_learning():
    nn = MinistNeuralNetwork()
    nn.train_neural_network()


def run_image_learning():
    data_loader = ImagesLoader()
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 01/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 02/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 03/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 04/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 05/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 06/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 07/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 08/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 09/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25/osoba 10/*.bmp")

    tender, number_of_classes = data_loader.get_data()
    train_x, train_y, test_x, test_y = DataFormatter.format_data(tender)
    nn = ImageNeuralNetwork(train_x, train_y, test_x, test_y, number_of_classes)
    nn.train_neural_network()

    print()
    print(number_of_classes)


if __name__ == "__main__":
    run_image_learning()
