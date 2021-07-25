# generator : 输入层维数：128（特征维数）+20（噪声维数）   隐层数：256  输出层：128
# subsititude detector: 128 - 256 - 1

from keras.layers import Input, Dense, Activation
from keras.layers.merge import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam



from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, svm, tree
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np
from VOTEClassifier import VOTEClassifier


class MalGAN:
    def __init__(
        self,
        blackbox="RF",
        same_train_data=1,
        filename="mydata.npz",
        apifeature_dims=160,
    ):
        self.apifeature_dims = apifeature_dims
        self.z_dims = 20  # The larger the noise length is, the more API is added
        self.hide_layers = 256
        self.generator_layers = [
            self.apifeature_dims + self.z_dims,
            self.hide_layers,
            self.apifeature_dims,
        ]
        self.substitute_detector_layers = [self.apifeature_dims, self.hide_layers, 1]
        self.blackbox = blackbox  # RF LR DT SVM MLP VOTE
        self.same_train_data = same_train_data  # MalGAN and the black-boxdetector are trained on same or different training sets
        optimizer = Adam(lr=0.001)
        self.filename = filename

        # Build and Train blackbox_detector
        self.blackbox_detector = self.build_blackbox_detector()

        # Build and compile the substitute_detector
        self.substitute_detector = self.build_substitute_detector()
        self.substitute_detector.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes malware and noise as input and generates adversarial malware examples
        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        input = [example, noise]
        malware_examples = self.generator(input)

        # For the combined model we will only train the generator
        self.substitute_detector.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.substitute_detector(malware_examples)

        # The combined model  (stacked generator and substitute_detector)
        # Trains the generator to fool the discriminator
        self.combined = Model(input, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_blackbox_detector(self):

        if self.blackbox is "RF":
            blackbox_detector = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=1
            )
        elif self.blackbox is "SVM":
            blackbox_detector = svm.SVC()
        elif self.blackbox is "LR":
            blackbox_detector = linear_model.LogisticRegression()
        elif self.blackbox is "DT":
            blackbox_detector = tree.DecisionTreeClassifier()
        elif self.blackbox is "MLP":
            blackbox_detector = MLPClassifier(
                hidden_layer_sizes=(50,),
                max_iter=10,
                alpha=1e-4,
                solver="sgd",
                verbose=0,
                tol=1e-4,
                random_state=1,
                learning_rate_init=0.1,
            )
        elif self.blackbox is "VOTE":
            blackbox_detector = VOTEClassifier()
        elif self.blackbox is "XGB":
            blackbox_detector = XGBClassifier(max_depth=5, n_estimators=90)

        return blackbox_detector

    def build_generator(self):

        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation="sigmoid")(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name="generator")
        generator.summary()
        return generator

    def build_substitute_detector(self):

        input = Input(shape=(self.substitute_detector_layers[0],))
        x = input
        for dim in self.substitute_detector_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation="sigmoid")(x)
        substitute_detector = Model(input, x, name="substitute_detector")
        substitute_detector.summary()
        return substitute_detector

    def load_data(self):

        data = np.load(self.filename)
        xmal, ymal, xben, yben = data["xmal"], data["ymal"], data["xben"], data["yben"]
        return (xmal, ymal), (xben, yben)

    def train(self, epochs=500, batch_size=32, is_first=1):

        # Load and Split the dataset
        (xmal, ymal), (xben, yben) = self.load_data()
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(
            xmal, ymal, test_size=0.20
        )
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(
            xben, yben, test_size=0.20
        )
        if self.same_train_data:  # use the same train_data
            bl_xtrain_mal, bl_ytrain_mal, bl_xtrain_ben, bl_ytrain_ben = (
                xtrain_mal,
                ytrain_mal,
                xtrain_ben,
                ytrain_ben,
            )
        else:  # use the different train_data
            xtrain_mal, bl_xtrain_mal, ytrain_mal, bl_ytrain_mal = train_test_split(
                xtrain_mal, ytrain_mal, test_size=0.50
            )
            xtrain_ben, bl_xtrain_ben, ytrain_ben, bl_ytrain_ben = train_test_split(
                xtrain_ben, ytrain_ben, test_size=0.50
            )

        # if is_first is Ture, Train the blackbox_detctor
        if is_first:
            self.blackbox_detector.fit(
                np.concatenate([xmal, xben]), np.concatenate([ymal, yben])
            )

        ytrain_ben_blackbox = self.blackbox_detector.predict(bl_xtrain_ben)
        Original_Train_TPR = self.blackbox_detector.score(bl_xtrain_mal, bl_ytrain_mal)
        Original_Test_TPR = self.blackbox_detector.score(xtest_mal, ytest_mal)
        Train_TPR, Test_TPR = [Original_Train_TPR], [Original_Test_TPR]
        best_TPR = 1.0
        for epoch in range(epochs):

            for step in range(xtrain_mal.shape[0] // batch_size):

                # --------------------------
                #  Train substitute_detector
                # --------------------------

                # Select a random batch of malware examples
                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))
                idx = np.random.randint(0, xmal_batch.shape[0], batch_size)
                xben_batch = xtrain_ben[idx]
                yben_batch = ytrain_ben_blackbox[idx]

                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = self.blackbox_detector.predict(
                    np.ones(gen_examples.shape) * (gen_examples > 0.5)
                )

                # Train the substitute_detector
                d_loss_fake = self.substitute_detector.train_on_batch(
                    gen_examples, ymal_batch
                )
                d_loss_real = self.substitute_detector.train_on_batch(
                    xben_batch, yben_batch
                )
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))

                # Train the generator
                g_loss = self.combined.train_on_batch(
                    [xmal_batch, noise], np.zeros((batch_size, 1))
                )

            # Compute Train TPR
            noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtrain_mal, noise])
            TPR = self.blackbox_detector.score(
                np.ones(gen_examples.shape) * (gen_examples > 0.5), ytrain_mal
            )
            Train_TPR.append(TPR)

            # Compute Test TPR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            TPR = self.blackbox_detector.score(
                np.ones(gen_examples.shape) * (gen_examples > 0.5), ytest_mal
            )
            Test_TPR.append(TPR)

            # Save best model
            if TPR < best_TPR:
                self.combined.save_weights("saves/malgan.h5")
                best_TPR = TPR

            # Plot the progress
            if is_first:
                print(
                    "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                    % (epoch, d_loss[0], 100 * d_loss[1], g_loss)
                )

        flag = ["DiffTrainData", "SameTrainData"]
        print("\n\n---{0} {1}".format(self.blackbox, flag[self.same_train_data]))
        print(
            "\nOriginal_Train_TPR: {0}, Adver_Train_TPR: {1}".format(
                Original_Train_TPR, Train_TPR[-1]
            )
        )
        print(
            "\nOriginal_Test_TPR: {0}, Adver_Test_TPR: {1}".format(
                Original_Test_TPR, Test_TPR[-1]
            )
        )

        # Plot TPR
        plt.figure()
        plt.plot(
            range(len(Train_TPR)), Train_TPR, c="r", label="Training Set", linewidth=2
        )
        plt.plot(
            range(len(Test_TPR)),
            Test_TPR,
            c="g",
            linestyle="--",
            label="Validation Set",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(
            "saves/Epoch_TPR({0}, {1}).png".format(
                self.blackbox, flag[self.same_train_data]
            )
        )
        plt.show()

    def retrain_blackbox_detector(self):
        (xmal, ymal), (xben, yben) = self.load_data()
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(
            xmal, ymal, test_size=0.20
        )
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(
            xben, yben, test_size=0.20
        )
        # Generate Train Adversarial Examples
        noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
        gen_examples = self.generator.predict([xtrain_mal, noise])
        gen_examples = np.ones(gen_examples.shape) * (gen_examples > 0.5)
        self.blackbox_detector.fit(
            np.concatenate([xtrain_mal, xtrain_ben, gen_examples]),
            np.concatenate([ytrain_mal, ytrain_ben, ytrain_mal]),
        )

        # Compute Train TPR
        train_TPR = self.blackbox_detector.score(gen_examples, ytrain_mal)

        # Compute Test TPR
        noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
        gen_examples = self.generator.predict([xtest_mal, noise])
        gen_examples = np.ones(gen_examples.shape) * (gen_examples > 0.5)
        test_TPR = self.blackbox_detector.score(gen_examples, ytest_mal)
        print(
            "\n---TPR after the black-box detector is retrained(Before Retraining MalGAN)."
        )
        print("\nTrain_TPR: {0}, Test_TPR: {1}".format(train_TPR, test_TPR))


if __name__ == "__main__":
    malgan = MalGAN(blackbox="DT", filename="datanew60.npz", apifeature_dims=60)
    malgan.train(epochs=100, batch_size=32)
    malgan.retrain_blackbox_detector()
    malgan.train(epochs=100, batch_size=32, is_first=False)
