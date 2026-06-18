"""
Bootstrap confidence intervals on the held-out test set.

The phase scripts report single point estimates. A single fixed test set
(n=1,106) still carries sampling uncertainty, so this script puts a 95%
percentile-bootstrap interval around the headline predictive numbers, and -
more usefully - around the multimodal F1 *lift* (paired on the same resamples,
which cancels a lot of the shared variance).

We bootstrap two things:
  - the tabular baselines (RF/XGB/MLP), using the test predictions Phase 2
    already saved to data/baseline_test_predictions.npz; and
  - the multimodal (+DistilBERT) classifiers, which we rebuild here from
    Phase 9's own functions (frozen encoder + train-only PCA), so the point
    estimates match phase9_multimodal_bert.json exactly.

Outputs
  results/bootstrap_ci.json
  figures/bootstrap_ci.png
"""



from __future__ import annotations
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from common import (load_processed, load_raw, make_binary_target,
                    save_json, set_plot_style,
                    RESULTS_DIR, FIGURES_DIR, DATA_DIR, RANDOM_SEED)
from phase2_baselines import train_mlp, mlp_predict
from phase9_multimodal_bert import make_comment, encode_comments




N_BOOT = 5000






def build_multimodal_predictions(data):
    """Reconstruct Phase 9's multimodal test predictions (pred, score per model).

    Mirrors phase9 exactly - same split replay, frozen DistilBERT, train-only
    PCA + scaler, same classifier configs - so the point estimates line up with
    the committed phase9 JSON.
    """
    X_tab_train, X_tab_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]


    df = make_binary_target(load_raw())
    idx_train, idx_test, _, _ = train_test_split(
        np.arange(len(df)), df["y"].values, test_size=0.25,
        stratify=df["y"].values, random_state=RANDOM_SEED)
    comments = [make_comment(i, int(df["y"].iloc[i])) for i in range(len(df))]
    emb_train = encode_comments([comments[i] for i in idx_train])
    emb_test = encode_comments([comments[i] for i in idx_test])




    pca = PCA(n_components=32, random_state=RANDOM_SEED).fit(emb_train)
    scaler = StandardScaler().fit(pca.transform(emb_train))
    txt_train = scaler.transform(pca.transform(emb_train)).astype(np.float32)
    txt_test = scaler.transform(pca.transform(emb_test)).astype(np.float32)
    X_tr = np.concatenate([X_tab_train, txt_train], axis=1)
    X_te = np.concatenate([X_tab_test, txt_test], axis=1)




    npr = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    preds = {}
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1).fit(X_tr, y_train)
    preds["RandomForest"] = (rf.predict(X_te), rf.predict_proba(X_te)[:, 1])
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                        scale_pos_weight=npr, random_state=RANDOM_SEED,
                        n_jobs=-1, verbosity=0).fit(X_tr, y_train)
    preds["XGBoost"] = (xgb.predict(X_te), xgb.predict_proba(X_te)[:, 1])
    mlp = train_mlp(X_tr, y_train, class_weight=float(npr))
    preds["MLP"] = mlp_predict(mlp, X_te)
    return preds






def boot_indices(n, n_boot, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def ci(values):
    a = np.asarray([v for v in values if v == v])  # drop nan
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def metric_on(idx, y, pred, score, metric):
    yt = y[idx]
    if metric == "f1":
        return f1_score(yt, pred[idx], zero_division=0)
    try:
        return roc_auc_score(yt, score[idx])
    except ValueError:
        return float("nan")






def main():
    set_plot_style()
    data = load_processed()
    y_test = data["y_test"]
    n = len(y_test)

    npz = np.load(DATA_DIR / "baseline_test_predictions.npz")
    tab = {"RandomForest": (npz["rf_pred"], npz["rf_score"]),
           "XGBoost": (npz["xgb_pred"], npz["xgb_score"]),
           "MLP": (npz["mlp_pred"], npz["mlp_score"])}
    print("[bootstrap] rebuilding multimodal predictions ...")
    mm = build_multimodal_predictions(data)

    idxs = boot_indices(n, N_BOOT)
    out = {"n_test": int(n), "n_boot": N_BOOT, "tabular": {}, "multimodal": {}, "lift_f1": {}}
    for model in ["RandomForest", "XGBoost", "MLP"]:
        tp, ts = tab[model]
        mp, ms = mm[model]
        f1_tab = [metric_on(ix, y_test, tp, ts, "f1") for ix in idxs]
        f1_mm = [metric_on(ix, y_test, mp, ms, "f1") for ix in idxs]
        auc_tab = [metric_on(ix, y_test, tp, ts, "auc") for ix in idxs]
        auc_mm = [metric_on(ix, y_test, mp, ms, "auc") for ix in idxs]
        lift = [a - b for a, b in zip(f1_mm, f1_tab)]  # paired on same resample
        out["tabular"][model] = {
            "f1": float(f1_score(y_test, tp, zero_division=0)), "f1_ci": ci(f1_tab),
            "auc": float(roc_auc_score(y_test, ts)), "auc_ci": ci(auc_tab)}
        out["multimodal"][model] = {
            "f1": float(f1_score(y_test, mp, zero_division=0)), "f1_ci": ci(f1_mm),
            "auc": float(roc_auc_score(y_test, ms)), "auc_ci": ci(auc_mm)}
        lo, hi = ci(lift)
        out["lift_f1"][model] = {"lift": float(np.mean(lift)), "ci": (lo, hi),
                                 "excludes_zero": bool(lo > 0)}
        print(f"  {model:13s} tab F1 {out['tabular'][model]['f1']:.3f} "
              f"-> mm F1 {out['multimodal'][model]['f1']:.3f}  "
              f"lift {np.mean(lift):+.3f} [{lo:+.3f},{hi:+.3f}]")
    save_json(out, RESULTS_DIR / "bootstrap_ci.json")


    # Left: F1 with 95% CI (tabular vs multimodal). Right: the paired F1 lift + CI.
    models = ["RandomForest", "XGBoost", "MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(models))
    for j, (kind, colour, dx) in enumerate([("tabular", "#95a5a6", -0.15),
                                            ("multimodal", "#2980b9", 0.15)]):
        f1s = [out[kind][m]["f1"] for m in models]
        los = [out[kind][m]["f1"] - out[kind][m]["f1_ci"][0] for m in models]
        his = [out[kind][m]["f1_ci"][1] - out[kind][m]["f1"] for m in models]
        axes[0].errorbar(x + dx, f1s, yerr=[los, his], fmt="o", capsize=4,
                         color=colour, label=kind)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_ylabel("F1"); axes[0].set_title("Test F1 with 95% bootstrap CI")
    axes[0].legend(fontsize=8)
    lifts = [out["lift_f1"][m]["lift"] for m in models]
    llo = [lifts[i] - out["lift_f1"][m]["ci"][0] for i, m in enumerate(models)]
    lhi = [out["lift_f1"][m]["ci"][1] - lifts[i] for i, m in enumerate(models)]
    axes[1].errorbar(x, lifts, yerr=[llo, lhi], fmt="s", capsize=4, color="#27ae60")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, fontsize=9)
    axes[1].set_ylabel("F1 lift (multimodal - tabular)")
    axes[1].set_title("Multimodal F1 lift, paired 95% CI")
    fig.suptitle("Bootstrap CIs on the held-out test set (n=1,106)", fontweight="bold")
    fig.savefig(FIGURES_DIR / "bootstrap_ci.png"); plt.close(fig)
    print("[bootstrap] done")


if __name__ == "__main__":
    main()
