# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip, numpy, torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    batch_size = 5  # nombre de données lues à chaque fois
    nb_epochs = 10  # nombre de fois que la base de données sera lue
    eta = 0.00001  # taux d'apprentissage

    eta_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    results = []

    for index, eta in enumerate(eta_list):
        results.append([])
        # on lit les données
        ((data_train, label_train), (data_test, label_test)) = torch.load(
            gzip.open("mnist.pkl.gz")
        )
        # on crée les lecteurs de données
        train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
        test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )

        # on initialise le modèle et ses poids
        model = torch.nn.Linear(data_train.shape[1], label_train.shape[1])
        torch.nn.init.uniform_(model.weight, -0.001, 0.001)
        # on initiliase l'optimiseur
        loss_func = torch.nn.MSELoss(reduction="sum")
        optim = torch.optim.SGD(model.parameters(), lr=eta)

        for n in range(nb_epochs):
            # on lit toutes les données d'apprentissage
            for x, t in train_loader:
                # on calcule la sortie du modèle
                y = model(x)
                # on met à jour les poids
                loss = loss_func(t, y)
                loss.backward()
                optim.step()
                optim.zero_grad()

            # test du modèle (on évalue la progression pendant l'apprentissage)
            acc = 0.0
            # on lit toutes les donnéees de test
            for x, t in test_loader:
                # on calcule la sortie du modèle
                y = model(x)
                # on regarde si la sortie est correcte
                acc += torch.argmax(y, 1) == torch.argmax(t, 1)
            # on affiche le pourcentage de bonnes réponses
            print(acc / data_test.shape[0])
            acc = acc.numpy()
            acc = acc[0]
            results[index].append(acc / data_test.shape[0])

    x = range(1, nb_epochs + 1)
    plt.figure()
    for i, eta in enumerate(eta_list):
        accuracy = results[i]
        plt.plot(x, accuracy, label=eta)
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Number of epochs")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.xlim((1, 10))
    plt.show()

    plt.figure()
    plt.semilogx(eta_list, [max(accuracy) for accuracy in results], "o-")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS learning rate")
    plt.tight_layout()
    plt.show()
