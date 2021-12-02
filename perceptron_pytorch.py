# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
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

    # on lit les données
    ((data_train, label_train), (data_test, label_test)) = torch.load(
        gzip.open("mnist.pkl.gz")
    )

    eta_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    results = []

    for index, eta in enumerate(eta_list):
        print(f"eta : {eta}")
        results.append([])

        # on initialise le modèle et ses poids
        w = torch.empty((data_train.shape[1], label_train.shape[1]), dtype=torch.float)
        b = torch.empty((1, label_train.shape[1]), dtype=torch.float)
        torch.nn.init.uniform_(w, -0.001, 0.001)
        torch.nn.init.uniform_(b, -0.001, 0.001)

        nb_data_train = data_train.shape[0]
        nb_data_test = data_test.shape[0]
        indices = numpy.arange(nb_data_train, step=batch_size)
        for n in range(nb_epochs):
            # on mélange les (indices des) données
            numpy.random.shuffle(indices)
            # on lit toutes les données d'apprentissage
            for i in indices:
                # on récupère les entrées
                x = data_train[i : i + batch_size]
                # on calcule la sortie du modèle
                y = torch.mm(x, w) + b
                # on regarde les vrais labels
                t = label_train[i : i + batch_size]
                # on met à jour les poids
                grad = t - y
                w += eta * torch.mm(x.T, grad)
                b += eta * grad.sum(axis=0)

            # test du modèle (on évalue la progression pendant l'apprentissage)
            acc = 0.0
            # on lit toutes les donnéees de test
            for i in range(nb_data_test):
                # on récupère l'entrée
                x = data_test[i : i + 1]
                # on calcule la sortie du modèle
                y = torch.mm(x, w) + b
                # on regarde le vrai label
                t = label_test[i : i + 1]
                # on regarde si la sortie est correcte
                acc += torch.argmax(y, 1) == torch.argmax(t, 1)
            # on affiche le pourcentage de bonnes réponses
            # print(acc / nb_data_test)
            acc = acc.numpy()
            acc = acc[0]
            results[index].append(acc / nb_data_test)

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
