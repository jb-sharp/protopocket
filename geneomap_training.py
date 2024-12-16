"""
Training GENEOmap on ProSPECCTs data from eta vectors
@author: Giovanni Bocchi
@institution: University of Milan
@email: giovanni.bocchi1@unimi.it
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
from torch import nn

class GENEOmap(nn.Module):
    """
    Class representing GENEOmap model that takes as input 
    the vector eta and returns the modified embedding A*eta.
    Given two vectors eta1 and eta2 it computes the similarity 
    score between them.
    
    """

    def __init__(self, train_set, test_set = None, epochs = 30, b_size = 16, l_rate = 1e-4):
        """
        Initializes a GENEOmap object.

        Parameters
        ----------
        train : list
            Contains positive and negative training pairs.
        test : list, optional
            Contains positive and negative testing pairs.
        epochs : int, optional
            Total number of epochs. The default is 30.
        batch_size : int, optional
            Training batch size. The default is 16.
        learning_rate : float, optional
            The optimizer leraning rate. The default is 1e-4.

        """

        super().__init__()
        self.epochs = epochs
        self.batch_size = b_size
        self.learning_rate = l_rate
        f1_train = train_set[0].astype(np.float32)
        f2_train = train_set[1].astype(np.float32)
        gt_train = train_set[2].astype(np.float32)
        self.trained = False

        self.linear = torch.nn.Linear(f1_train.shape[1],
                                      f1_train.shape[1],
                                      bias = False)

        data_train = list(zip(f1_train, f2_train, gt_train))
        self.train_loader = torch.utils.data.DataLoader(dataset = data_train,
                                                      batch_size = self.batch_size,
                                                      shuffle = True)
        if test_set is not None:
            f1_test = test_set[0].astype(np.float32)
            f2_test = test_set[1].astype(np.float32)
            gt_test = test_set[2].astype(np.float32)
            data_test = list(zip(f1_test, f2_test, gt_test))
            self.test_loader = torch.utils.data.DataLoader(dataset = data_test)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr = self.learning_rate)
        self.activation = nn.Identity()
        self.criterion = nn.MSELoss()

    def forward(self, eta):
        """
        Computes the embedding A*eta

        Parameters
        ----------
        eta : tensor
              The input vectors eta.

        Returns
        -------
        tensor
            The embeddings A*eta.

        """
        return self.linear(eta)

    def score(self, x1, x2):
        """
        Compute the similarity score between two vectors eta1 and eta2-

        Parameters
        ----------
        eta1 : tensor
               First input vectors.
        eta2 : tensor
               Second input vectors.

        Returns
        -------
        tensor
            Similary scores between eta1 and eta2.

        """
        o1 = self(x1)
        o2 = self(x2)
        return ((torch.abs(o1 - o2)).amax(axis = 1)).reshape(x1.shape[0], 1)

    def train_model(self):
        """
        Trains the model.

        """

        train_losses = []
        params = []

        # early_stopping = EarlyStopping(tolerance = 20, min_delta = 0.00001)

        for epoch in range(self.epochs):

            ll = 0

            for (x1b, x2b, yb) in self.train_loader:
                # Predictions
                p = self.score(x1b, x2b)
                # Calculate Loss + Linf penality
                act = self.activation(p)
                loss = 2*self.criterion(act, yb)
                loss = loss + ((torch.abs(self.linear.weight).sum(axis = 1)).max() - 1.0)**2
                ll += loss.data
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            ll = ll / len(self.train_loader)

            train_losses.append(ll.detach().cpu())
            params.append(self.state_dict())

            if hasattr(self, "testLoader"):
                with torch.no_grad():

                    tel = 0.0
                    for (x1b, x2b, yb) in self.test_loader:
                        # Predictions
                        p = self.score(x1b, x2b)
                        # Calculate Loss + Linf penality
                        act = self.activation(p)
                        loss = 2*self.criterion(act, yb)
                        loss = loss + ((torch.abs(self.linear.weight).sum(axis = 1)).max() - 1.0)**2
                        tel += loss.data

                    tel = tel / len(self.test_loader)

            if epoch % 100 == 0:
                if hasattr(self, "testLoader"):
                    string = f'epoch [{epoch + 1}/{self.epochs}], '
                    string += 'train loss: {ll:.5f} test loss: {tel:.5f}'
                    print(string)
                else:
                    print(f'epoch [{epoch + 1}/{self.epochs}], train loss: {ll:.5f}')

            ## early stopping
            # early_stopping(ll)
            # if early_stopping.early_stop:
            #   print("We are at epoch:", epoch)
            #   self.load_state_dict(params[np.argmin(train_losses)])
            #   self.trained = True
            #   break

        self.load_state_dict(params[np.argmin(train_losses)])
        self.trained = True

if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    prefix = os.getcwd()

    # ProSPECCTs datasets
    datasets = ["identical_structures",
                "identical_structures_similar_ligands", 
                "NMR_structures", 
                "decoy_shape_structures", 
                "decoy_rational_structures", 
                "kahraman_structures", 
                "barelier_structures",
                "review_structures"]

    EPOCHS = 600    # Epochs number
    SIZE1  = 500    # Size of positive pairs to sample
    SIZE2  = 500    # Size of negative pairs to sample

    for SEED in range(130, 181):

        if len(glob.glob(f"{prefix}/model-ounit*-ts1-{SIZE1}-ts2-{SIZE2}-mauc*-s{SEED}.dat")) < 1:

            # Setting seed for reproducibility
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            x1tr = []
            x2tr = []
            ytr = []
            x1te = []
            x2te = []
            yte = []

            train = []
            test = []
            leng = []

            print(f"SEED: {SEED}")

            for i, dataset in enumerate(datasets):

                # Loads eta vectors and groud truths
                path1 = f"{prefix}/data/geneomap_{dataset}.csv"
                path2 = f"{prefix}/prospectts/{dataset}.csv"

                df = pd.read_csv(path1, header = 0, index_col = 0)
                scaler = StandardScaler()
                xscaled = scaler.fit_transform(df)
                dfscaled = pd.DataFrame(xscaled, columns = df.columns)
                dfscaled.index = df.index

                truth = pd.read_csv(path2, header = None)
                truth.columns = ["P1", "P2", "label"]
                leng.append(len(truth))

                try:
                    yn  = (truth.label == "inactive").to_numpy()
                    y = yn.astype(np.float32).reshape((-1, 1)).astype(np.float32)
                    f1 = dfscaled.loc[truth.P1].to_numpy().astype(np.float32)
                    f2 = dfscaled.loc[truth.P2].to_numpy().astype(np.float32)
                except KeyError:
                    sys.exit()

                indices = np.arange(len(y))
                train_idx, test_idx = train_test_split(indices, train_size = 0.8, stratify = y)

                train_idx0 = [ix for ix in train_idx if y[ix] == 0]
                train_idx1 = [ix for ix in train_idx if y[ix] == 1]
                test_idx0 = [ix for ix in test_idx if y[ix] == 0]
                test_idx1 = [ix for ix in test_idx if y[ix] == 1]

                idx0_tr = np.random.choice(train_idx0, size = SIZE1, replace = True)
                idx1_tr = np.random.choice(train_idx1, size = SIZE2, replace = True)
                idx0_te = [i for i in range(len(y)) if (i not in idx0_tr) and (y[i] == 0)]
                idx1_te = [i for i in range(len(y)) if (i not in idx1_tr) and (y[i] == 1)]

                train.append((idx0_tr, idx1_tr))
                test.append((idx0_te, idx1_te))

                x1_train = np.concatenate([f1[idx0_tr],f1[idx1_tr]])
                x2_train = np.concatenate([f2[idx0_tr],f2[idx1_tr]])
                y_train = np.concatenate( [y[idx0_tr],  y[idx1_tr]])

                x1_test = np.concatenate([f1[idx0_te],f1[idx1_te]])
                x2_test = np.concatenate([f2[idx0_te],f2[idx1_te]])
                y_test = np.concatenate( [y[idx0_te],  y[idx1_te]])

                x1tr.append(x1_train)
                x2tr.append(x2_train)
                ytr.append(y_train)

                x1te.append(x1_test)
                x2te.append(x2_test)
                yte.append(y_test)

            f1tr = np.concatenate(x1tr)
            f2tr = np.concatenate(x2tr)
            ytr = np.concatenate(ytr)

            f1te = np.concatenate(x1te)
            f2te = np.concatenate(x2te)
            yte = np.concatenate(yte)

            # Instantiate GENEOnet model using training data
            model = GENEOmap((f1tr, f2tr, ytr),
                              #(f1te, f2te, yte),
                              epochs = EPOCHS)

            # Trains the model
            model.train_model()

            aucs = []
            TESTBOOL = True    # If True tests only on the validation parts
                               # otherwise on entire datasets
            for i, dataset in enumerate(datasets):
                path1 = f"{prefix}/data/geneomap_{dataset}.csv"
                path2 = f"{prefix}/prospectts/{dataset}.csv"

                df = pd.read_csv(path1, header = 0, index_col = 0)
                scaler = StandardScaler()
                xscaled = scaler.fit_transform(df)
                dfscaled = pd.DataFrame(xscaled, columns = df.columns)
                dfscaled.index = df.index

                truth = pd.read_csv(path2, header = None)
                truth.columns = ["P1", "P2", "label"]

                try:
                    yn = (truth.label == "inactive").to_numpy()
                    y = yn.astype(np.float32).reshape((-1, 1)).astype(np.float32)
                    f1 = dfscaled.loc[truth.P1].to_numpy().astype(np.float32)
                    f2 = dfscaled.loc[truth.P2].to_numpy().astype(np.float32)
                except KeyError:
                    sys.exit()

                if TESTBOOL:
                    i0te, i1te = test[i]
                    ite = np.concatenate([i0te, i1te])
                else:
                    i0te = np.where(y == 0)[0]
                    i1te = np.where(y == 1)[0]
                    ite = np.arange(len(y))

                s = model.score(torch.as_tensor(f1[ite]), torch.as_tensor(f2[ite]))
                a = model.activation(s).detach().cpu()
                fpr, tpr, thresholds = metrics.roc_curve(y[ite], a, pos_label = 1)
                auc = metrics.auc(fpr, tpr)
                aucs.append(auc)

                print(f"Dataset: {dataset} - AUC: {auc:.3f} TE0 {len(i0te)} TE1 {len(i1te)}")

            # Saves current model in the models folder
            ma = np.mean(aucs)
            if model.trained:

                if not os.path.isdir(f"{prefix}/models"):
                    os.mkdir(f"{prefix}/models")

                unit = model.linear.weight.shape[1]
                mname = f"{prefix}/models/model-ounit{unit}-ts1-{SIZE1}-ts2-{SIZE2}"
                mname += f"-mauc{ma:.2f}-s{SEED}.dat"
                torch.save({'epoch': 600,
                            'train_set': train,
                            'test_set': test,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': model.optimizer.state_dict(),
                            'aucs': aucs,
                            'mauc': ma,
                            'seed':SEED
                            },
                    mname)
            