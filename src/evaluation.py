from sklearn.model_selection import StratifiedKFold
import numpy as np

def evaluate_on_test(model, test_df, target_col):
    test_df = test_df.copy()
    # norm as string
    for c in test_df.columns:
        test_df[c] = test_df[c].astype(str).str.strip()
    preds = []
    real = test_df[target_col].tolist()
    for i in range(len(test_df)):
        preds.append(model.predict_row(test_df.iloc[i]))
    correct = sum(1 for p, r in zip(preds, real) if p == r)
    return correct / len(real)

def summarize_cv(df, target_col="lenses", k=5):
    from r1_model import OneRClassifier
    from id3_model import ID3Classifier
    from naive_bayes_model import NaiveBayesClassifier
    
    print("\n===============================")
    print(f" {k}-Fold Cross Validation Summary")
    print("===============================")
    print(f"{'Model':12s} {'Mean Acc':>10s} {'Std Dev':>10s}   Folds")

    models = [("1R", OneRClassifier), ("ID3", ID3Classifier), ("NaiveBayes", NaiveBayesClassifier)]

    for name, cls in models:
        scores = stratified_kfold_scores(cls, df, target_col, k=k)
        print(f"{name:12s} {np.mean(scores):10.3f} {np.std(scores):10.3f}   {', '.join(f'{s:.3f}' for s in scores)}")

def stratified_kfold_scores(model_cls, df, target_col, k=5, random_state=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    y = df[target_col].astype(str).values
    X = df.reset_index(drop=True)

    scores = []
    for train_idx, test_idx in skf.split(X, y):
        train_df = X.iloc[train_idx].copy()
        test_df  = X.iloc[test_idx].copy()

        model = model_cls()
        model.set_training_data(train_df)
        model.fit(target_col)

        acc = evaluate_on_test(model, test_df, target_col)
        scores.append(acc)
    return scores