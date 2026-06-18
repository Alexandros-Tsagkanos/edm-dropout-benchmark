"""
Co-Forest (Li & Zhou, 2007) - a faithful reimplementation.

Co-Forest ("Co-training Random Forest") is the ensemble self-labelled SSL
method that the two EDM anchor papers of this project find to be among the
strongest under label scarcity (Raftopoulos et al. 2025, MOOC dropout;
Kostopoulos et al. 2025, TVAE-SSL). Those studies run it from the `sslearn`
library; that package is not a dependency here, so we reimplement the
algorithm directly from the original paper to keep the project self-contained.

The mechanism (Li, M.; Zhou, Z.-H. "Improve Computer-Aided Diagnosis with
Machine Learning Techniques Using Undiagnosed Samples", IEEE TSMC-A 2007):

  * Train N random trees on bootstrap samples of the labelled set L.
  * Each round, for every tree i, its *companion ensemble* H_i (the other
    N-1 trees) pseudo-labels the unlabelled pool U. An unlabelled point is a
    candidate only when the companion ensemble's agreement (its averaged
    class probability) exceeds a confidence threshold theta.
  * Injection is error-controlled: tree i only accepts new pseudo-labels when
    the companion ensemble's error on L has decreased, and the number injected
    is capped so the *weighted* pseudo-label mass cannot outgrow the previous
    round's error-weighted mass (the e_{t-1} W_{t-1} < e_t W_t bound). This is
    what stops the classic self-training failure of runaway error propagation.
  * Tree i is retrained on L plus its accepted pseudo-labels (confidence used
    as sample weight). Iterate until no tree changes; predict by majority vote.

This is a faithful reimplementation of the published algorithm, not a
bit-exact port of any one library; `ssl_baseline.py` validates empirically
that it behaves as Co-Forest should (it beats a supervised model trained on
the labelled seed alone and approaches the fully-supervised upper bound as the
labelled ratio grows).
"""
from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state


class CoForest:
    """Co-Forest self-labelled ensemble.

    Parameters mirror the configuration the MOOC anchor paper used for
    Co-Forest (a Decision Tree base, n_estimators = 7, confidence threshold
    0.75).
    """

    def __init__(self, n_trees: int = 7, theta: float = 0.75,
                 max_iter: int = 15, random_state: int = 42):
        self.n_trees = n_trees
        self.theta = theta
        self.max_iter = max_iter
        self.random_state = random_state

    # -- helpers -------------------------------------------------------------
    def _proba_aligned(self, tree, X) -> np.ndarray:
        """predict_proba aligned to self.classes_ (a tree's bootstrap may miss
        a class, in which case its own classes_ is shorter; we pad with zeros)."""
        p = tree.predict_proba(X)
        if p.shape[1] == len(self.classes_):
            # fast path: same class ordering as the global one
            if np.array_equal(tree.classes_, self.classes_):
                return p
        out = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for j, c in enumerate(tree.classes_):
            out[:, np.searchsorted(self.classes_, c)] = p[:, j]
        return out

    def _companion(self, idx: int, X):
        """Averaged probability of every tree except `idx`; returns the
        predicted class and the agreement confidence (max averaged prob)."""
        probs = np.mean(
            [self._proba_aligned(self.trees_[j], X)
             for j in range(self.n_trees) if j != idx],
            axis=0,
        )
        pred = self.classes_[np.argmax(probs, axis=1)]
        conf = probs.max(axis=1)
        return pred, conf

    def _companion_error(self, idx: int, Lx, Ly) -> float:
        pred, _ = self._companion(idx, Lx)
        return max(float(np.mean(pred != Ly)), 1e-6)   # clamp away from 0

    # -- fit / predict -------------------------------------------------------
    def fit(self, Lx, Ly, Ux):
        rs = check_random_state(self.random_state)
        Lx = np.asarray(Lx); Ly = np.asarray(Ly); Ux = np.asarray(Ux)
        self.classes_ = np.unique(Ly)
        n = len(Ly)

        # N trees on bootstrap samples of the labelled set
        self.trees_ = []
        for _ in range(self.n_trees):
            boot = rs.randint(0, n, n)
            t = DecisionTreeClassifier(class_weight="balanced",
                                       random_state=rs.randint(0, 2**31 - 1))
            t.fit(Lx[boot], Ly[boot])
            self.trees_.append(t)

        if len(Ux) == 0:
            return self

        e_prev = np.full(self.n_trees, 0.5)
        W_prev = np.zeros(self.n_trees)

        for _ in range(self.max_iter):
            pseudo = [None] * self.n_trees
            # 1) every companion ensemble proposes pseudo-labels (using the
            #    current trees, so the round is internally consistent)
            for i in range(self.n_trees):
                e_i = self._companion_error(i, Lx, Ly)
                if e_i >= e_prev[i]:
                    continue
                pred, conf = self._companion(i, Ux)
                mask = conf > self.theta
                if not mask.any():
                    continue
                Xn, yn, wn = Ux[mask], pred[mask], conf[mask]
                Wi = float(wn.sum())
                # error-reduction cap (skipped the first productive round, when
                # W_prev == 0 and there is nothing yet to outgrow)
                if W_prev[i] > 0 and e_prev[i] * W_prev[i] < e_i * Wi:
                    limit = int(e_prev[i] * W_prev[i] / e_i)
                    if limit < len(yn):
                        keep = np.argsort(-wn)[:max(limit, 1)]
                        Xn, yn, wn = Xn[keep], yn[keep], wn[keep]
                        Wi = float(wn.sum())
                pseudo[i] = (Xn, yn, wn, e_i, Wi)

            # 2) retrain every tree that accepted new pseudo-labels
            changed = False
            new_trees = list(self.trees_)
            for i in range(self.n_trees):
                if pseudo[i] is None:
                    continue
                Xn, yn, wn, e_i, Wi = pseudo[i]
                Xtr = np.vstack([Lx, Xn])
                ytr = np.concatenate([Ly, yn])
                sw = np.concatenate([np.ones(len(Ly)), wn])
                t = DecisionTreeClassifier(class_weight="balanced",
                                       random_state=rs.randint(0, 2**31 - 1))
                t.fit(Xtr, ytr, sample_weight=sw)
                new_trees[i] = t
                e_prev[i] = e_i
                W_prev[i] = Wi
                changed = True

            self.trees_ = new_trees
            if not changed:
                break
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.mean([self._proba_aligned(t, X) for t in self.trees_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
