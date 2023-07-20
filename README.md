# Classification de chiffres avec modèle LeNet-4  (Boosté)

Ce projet démontre comment utiliser un modèle LeNet-4 (boosté) pour classifier les chiffres à partir de l'ensemble de données MNIST  (ou vos propres data) en utilisant TensorFlow et Keras. L'architecture LeNet-4 est implémentée dans `LeNet4.py`, et les fonctions d'entraînement et d'évaluation sont définies dans `training.py`.

## Prérequis

Avant d'exécuter le code, assurez-vous d'avoir installé les packages suivants :

- TensorFlow
- Keras
- NumPy
- pandas

## Définition des Fonctions

`load_process_data_MNIST()`

Cette fonction est responsable du chargement et du traitement du jeu de données MNIST, qui est un jeu de données couramment utilisé pour les chiffres écrits à la main.

```python
def load_process_data_MNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train, x_test, y_test)
```

- Elle utilise la fonction mnist.load_data() de TensorFlow pour charger le jeu de données MNIST dans x_train, y_train, x_test et y_test.

- Les données sont ensuite normalisées en divisant par 255.0 pour mettre les valeurs des pixels entre 0 et 1.

- Les étiquettes y_train et y_test sont encodées en "one-hot" à l'aide de keras.utils.to_categorical() pour les convertir en une représentation matricielle binaire.

- La fonction retourne (x_train, y_train, x_test, y_test).

`load_process_data(path_data_train, path_data_test)`

Cette fonction est responsable du chargement et du traitement de données personnalisées.


```python
def load_process_data(path_data_train, path_data_test):
    train_data = pd.read_csv(path_data_train, header=None)
    test_data = pd.read_csv(path_data_test, header=None)

    x_train, y_train = train_data.loc[:, 1:], train_data.loc[:, 0]
    y_train = keras.utils.to_categorical(y_train, 10)
    x_train = x_train.values.reshape(-1, 28, 28)
     
    x_test, y_test = test_data.loc[:, 1:], test_data.loc[:, 0]
    y_test = keras.utils.to_categorical(y_test, 10)
    x_test = x_test.values.reshape(-1, 28, 28)
    return (x_train, y_train, x_test, y_test)
```

- Elle lit les fichiers CSV spécifiés par path_data_train et path_data_test à l'aide de pd.read_csv().

- Les caractéristiques et les étiquettes sont séparées des données CSV et stockées dans x_train, y_train, x_test et y_test.

- Les étiquettes y_train et y_test sont encodées en "one-hot" en utilisant keras.utils.to_categorical().

- Les caractéristiques x_train et x_test sont remodelées en un format approprié pour être utilisées dans les couches Conv2D du réseau de neurones.

- à noter que je veux propose un petit dataset dans le dossier dataset, mais un dataset complet est disponible à l'adresse kaggle [dataset complet](https://www.kaggle.com/terromyann/datasets) (voir digit-number)

`model_LeNet4()`

Cette fonction crée et retourne le modèle LeNet4, une architecture de réseau de neurones convolutifs.

```python
def model_LeNet4():
    return keras.Sequential([
            layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
            layers.Conv2D(filters=20, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=50, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(units=500, activation="relu"),
            layers.Dense(units=10, activation="softmax")])
```

- Le modèle LeNet4 est créé à l'aide de keras.Sequential(), qui est une pile linéaire de couches.

- La première couche est une couche de remaniement (Reshape) qui adapte les dimensions de l'entrée (28, 28) à (28, 28, 1) pour correspondre à un canal d'image unique.

- Suivent deux couches Conv2D avec activation ReLU, suivies chacune d'une couche MaxPooling2D pour réduire la taille des cartes d'activation.
- Une couche d'aplatissement (Flatten) est ensuite utilisée pour aplatir les données afin de les préparer pour les couches de neurones entièrement connectées.

- Le modèle se termine par deux couches Dense (entièrement connectées) : une couche avec 500 neurones et activation ReLU, et une couche de sortie avec 10 neurones (correspondant aux 10 classes de sortie) avec activation softmax.

`boosted_training(model, x_train, y_train, n_iter=10)`

Cette fonction effectue un apprentissage par renforcement du modèle spécifié, en utilisant un ensemble de modèles faibles pour créer un modèle fort.

```python
def boosted_training(model, x_train, y_train, n_iter=10):
    models = []
    weights = np.ones(len(x_train)) / len(x_train)
    for _ in range(n_iter):
        model.fit(x_train, y_train, batch_size=128, epochs=5,
                  verbose=1, sample_weight=weights)
        models.append(model)
        predictions = model.predict(x_train)
        incorrect_predictions = np.argmax(predictions, axis=1) != np.argmax(y_train, axis=1)
        error = np.mean(incorrect_predictions)
        if error == 0:
            alpha = 0
        else:
            error = np.clip(error, 1e-15, 1 - 1e-15)
            alpha = 0.5 * np.log((1.0 - error) / error)
        weights *= np.exp(alpha * incorrect_predictions)
        weights /= np.sum(weights)
    return models
```

- La fonction prend en entrée le modèle model, les données d'entraînement x_train et y_train, ainsi que le nombre d'itérations n_iter (par défaut 10).

- Elle initialise une liste vide models pour stocker les modèles faibles.
Les poids weights sont initialisés uniformément pour chaque échantillon d'entraînement, ce qui signifie que chaque échantillon a une importance égale.

- Elle itère sur n_iter itérations, ajustant le modèle sur les données d'entraînement en utilisant les poids actuels.

- Ensuite, elle calcule les prédictions du modèle sur les données d'entraînement et identifie les prédictions incorrectes.

- Le poids est mis à jour pour donner plus d'importance aux échantillons mal classés et moins d'importance aux échantillons correctement classés.

- Après chaque itération, le modèle mis à jour est ajouté à la liste models.
La fonction retourne la liste des modèles mis à jour.


`save_boosted_model(models, path_to_save="./boosted_model/", model_name="model")` 

Cette fonction sauvegarde les modèles mis à jour après le processus d'apprentissage par renforcement.

```python
def save_boosted_model(models, path_to_save="./boosted_model/", model_name="model"):
    for i, model in enumerate(models):
        model.save(f"{path_to_save}{model_name}_{i}.h5")
```

- La fonction prend en entrée la liste des modèles models, le chemin d'accès de sauvegarde path_to_save (par défaut "./boosted_model/") et le nom de base du modèle model_name (par défaut "model").

- Elle itère sur chaque modèle mis à jour dans la liste models.

- Chaque modèle est sauvegardé en utilisant la fonction model.save() de Keras, avec un nom de fichier basé sur le chemin de sauvegarde, le nom de base du modèle et son index dans la liste.

## Utilisation du Code

Le code se termine avec une utilisation pratique des fonctions définies :

1. Utilisation du jeu de données MNIST avec le modèle LeNet4 :

```python
if __name__ == "__main__":
    # Chargement et traitement des données MNIST (LeNet4)
    (x_train, y_train, x_test, y_test) = load_process_data_MNIST()
    model = model_LeNet4()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)
    model.save("model.h5")
```

- Les données MNIST sont chargées et traitées à l'aide de load_process_data_MNIST().

- Le modèle LeNet4 est créé en utilisant model_LeNet4().

- Le modèle est compilé avec une fonction de perte, un optimiseur et des métriques.

- L'apprentissage est effectué sur les données d'entraînement avec model.fit().

- Le modèle entraîné est ensuite sauvegardé dans un fichier "model.h5".

2. Utilisation de vos propres données avec un modèle LeNet4 renforcé :

```python
if __name__ == "__main__":
    # Utilisation de vos propres données
    path_data_train = "chemin_vers_vos_donnees_train"
    path_data_test = "chemin_vers_vos_donnees_test"
    (x_train, y_train, x_test, y_test) = load_process_data(path_data_train, path_data_test)
    model_boosted = model_LeNet4()
    model_boosted.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    models = boosted_training(model_boosted, x_train, y_train, n_iter=5)
    save_boosted_model(models)
```

- Vous pouvez remplacer les chemins "chemin_vers_vos_donnees_train" et "chemin_vers_vos_donnees_test" par les chemins d'accès réels de vos données d'entraînement et de test.

- Les données personnalisées sont chargées et traitées à l'aide de load_process_data().

- Un nouveau modèle LeNet4 est créé pour le renforcement.

- Le modèle est compilé avec une fonction de perte, un optimiseur et des métriques.

- L'apprentissage par renforcement est effectué avec boosted_training() pour créer un ensemble de modèles mis à jour.

- Les modèles mis à jour sont sauvegardés à l'aide de save_boosted_model() dans le répertoire "boosted_model/" avec des noms de fichiers distincts pour chaque modèle.
