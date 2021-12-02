# IBI TP1 : Introduction à l'apprentissage profond



L'objectif de ce TP est d'implémenter l'algorithme du perceptron multi-couches vu en cours, de comprendre son fonctionnement et de prendre en main le framework PyTorch.



## Dataset

Nous utilisons le dataset MNIST contenant des images de taille (28*28) de chiffres manuscrits.



## Partie 1 : Perceptron

### Dimensions des tenseurs

La taille du tenseur `data_train` est de 63000 lignes (correspondant au nombre d'exemples) et 784 colonnes (correspondant à la dimension des données d'entrée). En effet, les images de taille 28*28 ont été déroulées en un vecteur de longueur 784.



Le tenseur `label_train` a une taille de 63000*10 car il y a nécéssairement autant de labels que d'exemples d'entraînement. Le 10 provient de la dimension de la sortie qui est un vecteur de taille 10, contenant la probabilité que la sortie soit égale à l'un des 10 chiffres.



La taille du tenseur `w` est de 784*10 puisque nous transformons nos données de taille 784 en une sortie égale à un vecteur de 10 classes (encodé en one-hot).



Le tenseur de biais  `b` est de taille 1*10 car il doit être de même taille que `xw` avec `x` un exemple tiré de `data_train` (matrice d'exemples d'apprentissage).



### Impact des hyperparamètres sur les performances

Nous avons fait varier `eta` de 10^-2 à 10^-10 afin d'observer l'impact sur les performances. Nous avons obtenu les courbes suivantes :



<img src="./images/AccuracyVSepoch.png" height="400" />

Cette figure montre bien qu'un taux d'apprentissage trop grand ne permet pas la convergence (cf `1e-2` où les mises à jour des poids sont trop grandes ce qui est assimilable à un modèle aléatoire) mais qu'un taux trop faible ne permet pas d'actualiser les poids lors de la descente de gradient (à partir de `1e-7` où nous observons une nette diminution des performances, et où pour des taux encore plus faibles, le modèle n'apprend simplement pas).



<img src="./images/AccuracyVSLR.png" height="400" />

Nous avons tracé ici la performance maximale atteinte en fonction du taux d'apprentissage en échelle log. Nous constatons que la valeur optimale du taux d'aprentissage se situe dans notre cas entre `1e-5` et `1e-3`.





## Partie 2 : Shallow Network





## Partie 3 : Deep Network





## Partie 4 : Pour aller plus loin (optionnel)





## Ressources

[Fonctions PyTorch]( https://pytorch.org/docs/stable/index.html)

[Tutoriel 1]( http://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

[Exemples introductifs]( http://pytorch.org/tutorials/beginner/deep_learning_ 60min_blitz.html)

