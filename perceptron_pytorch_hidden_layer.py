# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant juste les tenseurs)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip
import numpy
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    eta_list = [1e-2]
    results = {}
    number_units= 64

    for eta in eta_list:
        results_for_current_lr = []
        print(f"eta : {eta}")
        batch_size = 5  # nombre de données lues à chaque fois
        nb_epochs = 10  # nombre de fois que la base de données sera lue

        # on lit les données
        ((data_train, label_train), (data_test, label_test)
         ) = torch.load(gzip.open('mnist.pkl.gz'))

        # on initialise le modèle et ses poids
        w1 = torch.empty(
            (data_train.shape[1], number_units), dtype=torch.float)
        b1 = torch.empty((1, number_units), dtype=torch.float)

        w2 = torch.empty(
            (number_units, label_train.shape[1]), dtype=torch.float)
        b2 = torch.empty((1, label_train.shape[1]), dtype=torch.float)

        torch.nn.init.uniform_(w1, -0.001, 0.001)
        torch.nn.init.uniform_(b1, -0.001, 0.001)
        torch.nn.init.uniform_(w2, -0.001, 0.001)
        torch.nn.init.uniform_(b2, -0.001, 0.001)

        nb_data_train = data_train.shape[0]
        nb_data_test = data_test.shape[0]
        indices = numpy.arange(nb_data_train, step=batch_size)
        for n in range(nb_epochs):
            # on mélange les (indices des) données
            numpy.random.shuffle(indices)
            # on lit toutes les données d'apprentissage
            for i in indices:
                # on récupère les entrées
                x = data_train[i:i+batch_size]
                # on calcule la sortie du modèle
                y1 = 1 / (1+torch.exp(-(torch.mm(x, w1) + b1)))

                y2 = torch.mm(y1, w2) + b2

                # on regarde les vrais labels
                t = label_train[i:i+batch_size]
                # on met à jour les poids
                grad2 = (t-y2)

                grad1 = y1 * (1 - y1) * torch.mm(grad2, w2.T)

                w2 += eta * torch.mm(y1.T, grad2)
                b2 += eta * grad2.sum(axis=0)
                w1 += eta * torch.mm(x.T, grad1)
                b1 += eta * grad1.sum(axis=0)

            # test du modèle (on évalue la progression pendant l'apprentissage)
            acc = 0.
            # on lit toutes les donnéees de test
            for i in range(nb_data_test):
                # on récupère l'entrée
                x = data_test[i:i+1]
                # on calcule la sortie du modèle
                y1 = 1 / (1+torch.exp(-(torch.mm(x, w1) + b1)))
                y2 = torch.mm(y1, w2) + b2
                # on regarde le vrai label
                t = label_test[i:i+1]
                # on regarde si la sortie est correcte
                acc += torch.argmax(y2, 1) == torch.argmax(t, 1)
            # on affiche le pourcentage de bonnes réponses
            # print(acc/nb_data_test)
            acc = acc.numpy()/nb_data_test
            acc = acc[0]
            print(f"Accuracy : {acc}")
            results_for_current_lr.append(acc)
        results[eta] = results_for_current_lr



    x = range(1, nb_epochs + 1)
    plt.figure()
    for i, eta in enumerate(eta_list):
        accuracy = results[i]
        plt.plot(x, accuracy, label=unit)
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Number of epochs")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.xlim((1, 10))
    plt.show()

    plt.figure()
    plt.semilogx(eta_list, [max(accuracy) for accuracy in results], "o-")
    plt.xlabel("Number of units in hidden layer")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS number of units in hidden layer")
    plt.tight_layout()
    plt.show()
