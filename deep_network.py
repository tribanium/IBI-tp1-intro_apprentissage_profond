import torch, numpy, gzip
import matplotlib.pyplot as plt

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:cyan",
]


def init_weights(model, method, weight_scale):
    if isinstance(model, torch.nn.Linear):
        if method == "uniform":
            torch.nn.init.uniform_(model.weight, -weight_scale, weight_scale)
        elif method == "normal":
            torch.nn.init.normal_(model.weight, mean=0.0, std=weight_scale)
        elif method == "constant":
            torch.nn.init.constant_(model.weight, weight_scale)
        elif method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(model.weight)
        elif method == "xavier_normal":
            torch.nn.init.xavier_normal_(model.weight)
        else:
            raise Exception("Please use a valid weights init method.")


class NeuralNet:
    def __init__(
        self,
        batch_size,
        nb_epochs,
        eta,
        hidden_layers_units,
        optimizer,
        loss,
        init_method,
        weight_scale,
    ):
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.eta = eta
        self.hidden_layers_units = hidden_layers_units
        self.init_method = init_method
        self.weight_scale = weight_scale

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
        init_function = lambda model: init_weights(
            model, self.init_method, self.weight_scale
        )
        self.model.apply(init_function)

        # On initialise l'optimiseur
        self.loss_func = loss(reduction="sum")
        self.optim = optimizer(self.model.parameters(), lr=self.eta / self.batch_size)

    def run(self):

        print(100 * "=")
        print(f"Model : \n{self.model}\n")
        print(
            f"learning rate : {self.eta} | batch size : {self.batch_size} | init method : {self.init_method}"
        )
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
            print(
                f"Epoch {n+1}/{self.nb_epochs} \t|\t Test accuracy : {'{:.3f}'.format(acc / self.data_test.shape[0])} \t|\t Train accuracy : {'{:.3f}'.format(train_acc/self.data_train.shape[0])}"
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
        init_method="uniform",
        weight_scale=0.001,
    ) -> None:
        self.accuracy_eta = {"train": [], "test": []}
        self.accuracy_layers = {"train": [], "test": []}

        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.eta = eta
        self.hidden_layers_units = hidden_layers_units
        self.optimizer = optimizer
        self.loss = loss
        self.init_method = init_method
        self.weight_scale = weight_scale

    def test_eta(self):
        eta_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        print("ITERATING OVER LEARNING RATES...")
        for i, eta in enumerate(eta_list):
            print(f"\n### eta = {eta} ###\n")
            neuralnet = NeuralNet(
                self.batch_size,
                self.nb_epochs,
                eta,
                self.hidden_layers_units,
                self.optimizer,
                self.loss,
                self.init_method,
                self.weight_scale,
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
            # plt.plot(
            #     x, self.accuracy_eta["train"][i], color=colors[i], linestyle="dashed"
            # )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.title(
            f"LR iterations | batch size = {self.batch_size} | hidden layers : {', '.join(str(e) for e in self.hidden_layers_units)} |  weights init = {self.init_method}"
        )
        plt.show()

    def test_layers(self):
        nb_units = [512, 256, 64, 16]
        x = range(1, self.nb_epochs + 1)
        plt.figure()
        print("ITERATING OVER HIDDEN LAYERS UNITS...")
        color_index = 0
        for nb_layers in range(3, 1, -1):
            for i, unit in enumerate(nb_units[: 1 - nb_layers]):
                hidden_layers_units = [nb_units[i + k] for k in range(nb_layers)]
                neuralnet = NeuralNet(
                    self.batch_size,
                    self.nb_epochs,
                    self.eta,
                    hidden_layers_units,
                    self.optimizer,
                    self.loss,
                    self.init_method,
                    self.weight_scale,
                )
                neuralnet.run()

                if nb_layers == 3:
                    plt.plot(
                        x,
                        neuralnet.accuracy,
                        color=colors[color_index],
                        label="/".join(str(e) for e in hidden_layers_units),
                        linestyle="dashed",
                    )
                elif nb_layers == 2:
                    plt.plot(
                        x,
                        neuralnet.accuracy,
                        color=colors[color_index],
                        label="/".join(str(e) for e in hidden_layers_units),
                    )

                color_index += 1

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.title(
            f"nb of HL - nb units iterations | batch size = {self.batch_size} | LR = {'{:.0e}'.format(self.eta)}"
        )
        plt.show()

    # TODO : test initialisation des poids

    def test_weights_init(self):
        init_method_list = [
            "uniform",
            "normal",
            "constant",
            "xavier_uniform",
            "xavier_normal",
        ]

        # Plot stuff
        x = range(1, self.nb_epochs + 1)
        plt.figure()

        print("ITERATING OVER WEIGHT INIT METHODS...")
        for i, init_method in enumerate(init_method_list):
            neuralnet = NeuralNet(
                self.batch_size,
                self.nb_epochs,
                self.eta,
                self.hidden_layers_units,
                self.optimizer,
                self.loss,
                init_method,
                self.weight_scale,
            )
            neuralnet.run()

            plt.plot(
                x,
                neuralnet.accuracy,
                color=colors[i],
                label=init_method,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.title(
            f"Weights init (weight scale = {self.weight_scale}) | batch size = {self.batch_size} | LR = {'{:.0e}'.format(self.eta)} | HL : {', '.join(str(e) for e in self.hidden_layers_units)} "
        )
        plt.show()

    # TODO : test taille du mini batch

    def test_batch_size(self):
        batch_size_list = [1024, 512, 256, 128, 64, 32, 16, 8]

        # Plot stuff
        x = range(1, self.nb_epochs + 1)
        plt.figure()

        print("ITERATING OVER BATCH SIZE...")
        for i, batch_size in enumerate(batch_size_list):
            neuralnet = NeuralNet(
                batch_size,
                self.nb_epochs,
                self.eta,
                self.hidden_layers_units,
                self.optimizer,
                self.loss,
                self.init_method,
                self.weight_scale,
            )
            neuralnet.run()

            plt.plot(
                x,
                neuralnet.accuracy,
                color=colors[i],
                label=str(batch_size),
            )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim((1, self.nb_epochs))
        plt.legend()
        plt.title(
            f"BS iterations | weights init = {self.init_method} | LR = {'{:.0e}'.format(self.eta)} | HL : {', '.join(str(e) for e in self.hidden_layers_units)} "
        )
        plt.show()

    # TODO test cross entropy + softmax


# TODO : diminuer le learning rate quand on augmente le batch size
if __name__ == "__main__":
    batch_size = 32
    nb_epochs = 40
    eta = 1e-2
    hidden_layers_units = [512, 256]
    optimizer = torch.optim.SGD
    loss = torch.nn.MSELoss
    # loss = torch.nn.CrossEntropyLoss
    init_method = "xavier_uniform"
    weight_scale = 0.01

    # neuralnet = NeuralNet(
    #     batch_size=32,
    #     nb_epochs=nb_epochs,
    #     eta=eta,
    #     hidden_layers_units=hidden_layers_units,
    #     optimizer=optimizer,
    #     loss=loss,
    #     init_method=init_method,
    #     weight_scale=weight_scale,
    # )

    # neuralnet.run()
    # neuralnet.plot_accuracy()

    tests = Tests(
        batch_size=batch_size,
        nb_epochs=nb_epochs,
        eta=eta,
        hidden_layers_units=hidden_layers_units,
        optimizer=optimizer,
        loss=loss,
        init_method=init_method,
        weight_scale=weight_scale,
    )

    """ Testing learning rate """
    # tests.test_eta()

    """ Testing number of hidden layers and neurons """
    # tests.test_layers()

    """ Testing initial weigths """
    tests.test_weights_init()

    """ Testing mini batch size """
    # tests.test_batch_size()
