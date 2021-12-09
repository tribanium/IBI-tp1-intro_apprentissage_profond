import torch, numpy, gzip
import matplotlib.pyplot as plt

colors = ["tab:blue", "tab:orange", "tab:green"]


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, -0.001, 0.001)


class NeuralNet:
    def __init__(
        self,
        batch_size,
        nb_epochs,
        eta,
        hidden_layers_units=[],
        optimizer=torch.optim.SGD,
        loss=torch.nn.MSELoss,
    ):
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.eta = eta
        self.hidden_layers_units = hidden_layers_units

        self.accuracy = []
        self.train_accuracy = []

        self.nb_hidden_layers = len(hidden_layers_units)

        (
            (self.data_train, self.label_train),
            (self.data_test, self.label_test),
        ) = torch.load(gzip.open("mnist.pkl.gz"))

        # on crée les lecteurs de données
        self.train_dataset = torch.utils.data.TensorDataset(
            self.data_train, self.label_train
        )
        self.test_dataset = torch.utils.data.TensorDataset(
            self.data_test, self.label_test
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )

        # On initialise le modèle et ses poids
        layers_list = []
        if self.nb_hidden_layers > 0:
            layers_list.append(
                torch.nn.Linear(self.data_train.shape[1], hidden_layers_units[0])
            )
            for layer, layer_units in enumerate(hidden_layers_units):
                # Si on n'est pas à la dernière couche :
                if layer < self.nb_hidden_layers - 1:
                    layers_list.append(
                        torch.nn.Linear(
                            hidden_layers_units[layer], hidden_layers_units[layer + 1]
                        )
                    )
                    layers_list.append(torch.nn.ReLU())
                else:
                    layers_list.append(
                        torch.nn.Linear(
                            hidden_layers_units[layer], self.label_train.shape[1]
                        )
                    )
        else:
            layers_list.append(
                torch.nn.Linear(self.data_train.shape[1], self.label_train.shape[1])
            )

        self.model = torch.nn.Sequential(*layers_list)
        self.model.apply(init_weights)

        # On initialise l'optimiseur
        self.loss_func = loss(reduction="sum")
        self.optim = optimizer(self.model.parameters(), lr=self.eta)

    def run(self):
        print(f"Model : \n{self.model}\n")
        print(f"learning rate : {self.eta} | batch size : {self.batch_size}")
        print("Training model...\n")

        for n in range(self.nb_epochs):
            train_acc = 0.0
            # on lit toutes les données d'apprentissage
            for x, t in self.train_loader:
                # on calcule la sortie du modèle
                y = self.model(x)
                # on met à jour les poids
                loss = self.loss_func(t, y)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                train_acc += (torch.argmax(y, 1) == torch.argmax(t, 1)).numpy().sum()
            self.train_accuracy.append(train_acc / self.data_train.shape[0])

            # test du modèle (on évalue la progression pendant l'apprentissage)
            acc = 0.0
            # on lit toutes les donnéees de test
            for x, t in self.test_loader:
                # on calcule la sortie du modèle
                y = self.model(x)
                # on regarde si la sortie est correcte
                acc += torch.argmax(y, 1) == torch.argmax(t, 1)
            # on affiche le pourcentage de bonnes réponses
            acc = acc.numpy()
            acc = acc[0]
            if n % 10 == 9:
                print(
                    f"Epoch {n}/{self.nb_epochs} | Test accuracy : {acc / self.data_test.shape[0]} | Train accuracy : {train_acc/self.data_train.shape[0]} "
                )
            self.accuracy.append(acc / self.data_test.shape[0])

    def plot_accuracy(self):
        x = range(1, self.nb_epochs + 1)
        plt.figure()
        plt.plot(x, self.accuracy)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"LR {self.eta} | batch size = {self.batch_size} | hidden layers : {', '.join(str(e) for e in self.hidden_layers_units)}"
        )
        plt.xlim((1, self.nb_epochs))
        plt.show()


class Tests:
    def __init__(
        self,
        batch_size,
        nb_epochs,
        eta,
        hidden_layers_units=[],
        optimizer=torch.optim.SGD,
        loss=torch.nn.MSELoss,
    ) -> None:
        self.accuracy_eta = {"train": [], "test": []}
        self.accuracy_layers = {"train": [], "test": []}

        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.eta = eta
        self.hidden_layers_units = hidden_layers_units
        self.optimizer = optimizer
        self.loss = loss

    def test_eta(self, eta_list=None):
        print("ITERATING OVER LEARNING RATES...")
        for i, eta in enumerate(eta_list):
            print(f"### eta = {eta} ###\n")
            neuralnet = NeuralNet(
                self.batch_size,
                self.nb_epochs,
                eta,
                self.hidden_layers_units,
                self.optimizer,
                self.loss,
            )
            neuralnet.run()
            self.accuracy_eta["test"].append(neuralnet.accuracy)
            self.accuracy_eta["train"].append(neuralnet.train_accuracy)

        x = range(1, self.nb_epochs + 1)
        plt.figure()
        for i, eta in enumerate(eta_list):
            plt.plot(
                x,
                self.accuracy_eta["test"][i],
                color=colors[i],
                label="{:.0e}".format(eta),
            )
            plt.plot(
                x, self.accuracy_eta["train"][i], color=colors[i], linestyle="dashed"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.title(
            f"LR iterations | batch size = {self.batch_size} | hidden layers : {', '.join(str(e) for e in self.hidden_layers_units)}"
        )
        plt.show()

    def test_layers(self, nb_layers):
        print("ITERATING OVER HIDDEN LAYERS UNITS...")
        nb_units = [1000, 500, 250, 60, 30]
        plot_labels = []
        for i, unit in enumerate(nb_units[: 1 - nb_layers]):
            hidden_layers_units = [nb_units[i + k] for k in range(nb_layers)]
            neuralnet = NeuralNet(
                self.batch_size,
                self.nb_epochs,
                self.eta,
                hidden_layers_units,
                self.optimizer,
                self.loss,
            )
            neuralnet.run()
            self.accuracy_layers.append(neuralnet.accuracy)
            plot_labels.append("/".join(str(e) for e in hidden_layers_units))

        x = range(1, self.nb_epochs + 1)
        plt.figure()
        for i, eta in enumerate(nb_units[: 1 - nb_layers]):
            plt.plot(
                x,
                self.accuracy_layers["test"][i],
                color=colors[i],
                label=plot_labels[i],
            )
            plt.plot(
                x, self.accuracy_layers["train"][i], color=colors[i], linestyle="dashed"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.title(
            f"{nb_layers} HL - nb units iterations | batch size = {self.batch_size} | LR = {'{:.0e}'.format(self.eta)}"
        )
        plt.show()

    # TODO : test initialisation des poids

    # TODO : test taille du mini batch

    # TODO test cross entropy + softmax


if __name__ == "__main__":
    batch_size = 5
    nb_epochs = 50
    eta = 1e-3
    hidden_layers_units = [64, 32]
    optimizer = torch.optim.SGD
    # loss = torch.nn.CrossEntropyLoss

    neuralnet = NeuralNet(
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        eta=eta,
        hidden_layers_units=hidden_layers_units,
        optimizer=optimizer,
    )

    # neuralnet.run()
    # neuralnet.plot_accuracy()

    tests = Tests(
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        eta=eta,
        hidden_layers_units=hidden_layers_units,
        optimizer=optimizer,
    )

    """ Testing learning rate """
    # eta_list = [1e-2, 1e-3, 1e-4, 1e-5]
    # tests.test_eta(eta_list)

    """ Testing number of hidden layers and neurons """
    tests.test_layers(3)

    """ Testing initial weigths """

    """ Testing mini batch size """
