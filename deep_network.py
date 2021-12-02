import torch, numpy, gzip
import matplotlib.pyplot as plt


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

        self.accuracy = []

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

        # on initialise le modèle et ses poids
        layers_list = []
        if self.nb_hidden_layers > 0:
            layers_list.append(
                torch.nn.Linear(self.data_train.shape[1], hidden_layers_units[0])
            )
            for layer, layer_units in enumerate(hidden_layers_units):
                if layer < self.nb_hidden_layers - 1:
                    layers_list.append(
                        torch.nn.Linear(
                            hidden_layers_units[layer], hidden_layers_units[layer + 1]
                        )
                    )
                else:
                    layers_list.append(
                        torch.nn.Linear(
                            hidden_layers_units[layer], self.label_train.shape[1]
                        )
                    )
        self.model = torch.nn.Sequential(*layers_list)
        self.model.apply(self.init_weights)

        # on initialise l'optimiseur
        self.loss_func = loss(reduction="sum")
        self.optim = optimizer(self.model.parameters(), lr=self.eta)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.001, 0.001)

    def run(self):
        print(f"Model : \n{self.model}\n")
        print(f"learning rate : {self.eta} | batch size : {self.batch_size}")
        print("Training model...\n")
        for n in range(self.nb_epochs):
            # on lit toutes les données d'apprentissage
            for x, t in self.train_loader:
                # on calcule la sortie du modèle
                y = self.model(x)
                # on met à jour les poids
                loss = self.loss_func(t, y)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

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
            if n % 10 == 0:
                print(
                    f"Epoch {n}/{self.nb_epochs} - Accuracy : {acc / self.data_test.shape[0]}"
                )
            self.accuracy.append(acc / self.data_test.shape[0])

    def plot_accuracy(self):
        x = range(1, self.nb_epochs + 1)
        plt.figure()
        plt.plot(x, self.accuracy)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy VS Number of epochs")
        plt.tight_layout()
        plt.xlim((1, self.nb_epochs))
        plt.show()

    def test_eta(self, eta_list=None):
        print(f"Model : \n{self.model}\n")
        print("ITERATING OVER LEARNING RATES...")
        if not eta_list:
            raise Exception("Please specify a list of learning rates.")

        accuracy = []
        for i, eta in enumerate(eta_list):
            accuracy.append([])
            print(f"### eta = {eta} ###\n")
            optim = torch.optim.SGD(self.model.parameters(), lr=eta)

            for n in range(self.nb_epochs):
                # on lit toutes les données d'apprentissage
                for x, t in self.train_loader:
                    # on calcule la sortie du modèle
                    y = self.model(x)
                    # on met à jour les poids
                    loss = self.loss_func(t, y)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

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
                if n % 10 == 0:
                    print(
                        f"Epoch {n}/{self.nb_epochs} - Accuracy : {acc / self.data_test.shape[0]}"
                    )
                accuracy[i].append(acc / self.data_test.shape[0])
            print("\n")

        x = range(1, self.nb_epochs + 1)
        plt.figure()
        for i, eta in enumerate(eta_list):
            plt.plot(x, accuracy[i], label=eta)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy VS Number of epochs")
        plt.tight_layout()
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    batch_size = 5
    nb_epochs = 50
    eta = 1e-3
    hidden_layers_units = [256, 128, 64]
    optimizer = torch.optim.SGD
    loss = torch.nn.MSELoss

    neuralnet = NeuralNet(
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        eta=eta,
        hidden_layers_units=hidden_layers_units,
        optimizer=optimizer,
        loss=loss,
    )

    # neuralnet.run()
    # neuralnet.plot_accuracy()

    eta_list = [1e-2, 1e-3, 1e-4, 1e-5]
    neuralnet.test_eta(eta_list)
