from src.data.DataFormatter import DataFormatter
from src.images.ImagesLoader import ImagesLoader
from src.neuralnetwork.MultilayerPerceptronNeuralNetwork import MultilayerPerceptronNeuralNetwork
from src.neuralnetwork.NeuralNetworkMinistData import NeuralNetwork as MinistNeuralNetwork
from src.neuralnetwork.ConvolutionNeuralNetwork import ConvolutionNeuralNetwork

def run_minist_learning():
    nn = MinistNeuralNetwork()
    nn.train_neural_network()


def run_image_learning():
    data_loader = ImagesLoader()
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 01/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 02/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 03/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 04/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 05/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 06/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 07/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 08/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 09/*.bmp")
    data_loader.load_dara_and_add_as_class("../../Data/Kwadraty_25_szare/osoba 10/*.bmp")

    tender, number_of_classes = data_loader.get_data()
    accuracy_sum = 0
    time_sum = 0
    iterations = 1

    for n in range(0, iterations):
        train_x, train_y, test_x, test_y = DataFormatter.format_data(tender)
        nn = ConvolutionNeuralNetwork(train_x, train_y, test_x, test_y, number_of_classes, epoch=100)
        #nn = MultilayerPerceptronNeuralNetwork(train_x, train_y, test_x, test_y, number_of_classes, epoch=50)
        nn.train_neural_network()

        accuracy_sum += nn.accuracy
        time_sum += nn.time_of_training
        print(
            "ConvolutionNeuralNetwork accuracy: " + str(nn.accuracy) +
            ", time of training: " + str(nn.time_of_training) + " s ")

    average_time = time_sum/iterations
    average_accuracy = accuracy_sum/iterations
    print("Average accuracy: " + str(average_accuracy))
    print("Average time: " + str(average_time))


if __name__ == "__main__":
    run_image_learning()
