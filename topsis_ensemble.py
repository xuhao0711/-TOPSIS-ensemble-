"""Training-only TOPSIS, nested grouped CV and frozen weighted prediction."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CRITERIA = ['MAE', 'RMSE', 'R2']
DIRECTIONS = ['min', 'min', 'max']
DEFAULT_WEIGHTS = [0.5, 0.25, 0.25]


def topsis(table, weights=DEFAULT_WEIGHTS):
    """Column L2 normalization; negative R² remains signed; no reciprocal costs."""
    a = table[CRITERIA].to_numpy(float)
    w = np.asarray(weights, float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite CV criteria (check constant targets and folds).')
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Three finite nonnegative weights with positive sum required.')
    w = w / w.sum()
    norm = np.linalg.norm(a, axis=0)
    v = a / np.where(norm == 0, 1, norm) * w
    ideal = np.array([v[:, j].min() if d == 'min' else v[:, j].max()
                      for j, d in enumerate(DIRECTIONS)])
    worst = np.array([v[:, j].max() if d == 'min' else v[:, j].min()
                      for j, d in enumerate(DIRECTIONS)])
    dp = np.linalg.norm(v - ideal, axis=1)
    dm = np.linalg.norm(v - worst, axis=1)
    c = np.divide(dm, dp + dm, out=np.full(len(a), .5), where=(dp + dm) > 0)
    out = table.copy()
    out['distance_ideal'], out['distance_worst'], out['closeness'] = dp, dm, c
    out = out.sort_values(['closeness', 'model'], ascending=[False, True]).reset_index(drop=True)
    out['rank'] = np.arange(1, len(out) + 1)
    out['ensemble_weight'] = out.closeness / out.closeness.sum() if out.closeness.sum() else 1 / len(out)
    return out


def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)),
                RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=float(r2_score(y, p, force_finite=False)))


def assert_isolation(train, test, features):
    """Check global IDs/ancestry, provided source groups and identical feature vectors."""
    def ids(df):
        result = set()
        for col in ['row_id', 'parent_id_1', 'parent_id_2']:
            if col in df:
                values = pd.to_numeric(df[col], errors='raise').dropna()
                result.update(values[values >= 0].tolist())
        return result
    if 'row_id' not in train or 'row_id' not in test:
        raise ValueError('Global original row_id required; never number each sheet independently.')
    if train.row_id.isna().any() or test.row_id.isna().any():
        raise ValueError('Missing row_id.')
    if ids(train) & ids(test):
        raise ValueError('Train/test row or parent IDs overlap.')
    for col in ['group_id', 'source_id']:
        if col in train and col in test and set(train[col].dropna()) & set(test[col].dropna()):
            raise ValueError(f'Train/test {col} overlaps.')
    a = set(map(tuple, train[features].to_numpy()))
    b = set(map(tuple, test[features].to_numpy()))
    if a & b:
        raise ValueError('Identical feature rows cross train/test; group replicates before splitting.')
    return {'id_ancestry_overlap': False, 'feature_overlap': False,
            'train_rows': len(train), 'test_rows': len(test)}


class FoldAugmentedRegressor(RegressorMixin, BaseEstimator):
    """Augment ONLY the fit subset. Callback returns (X_aug, y_aug)."""
    def __init__(self, estimator, augment=None):
        self.estimator = estimator
        self.augment = augment

    def fit(self, X, y):
        if self.augment is not None:
            X, y = self.augment(np.asarray(X).copy(), np.asarray(y).copy())
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class FrozenEnsemble:
    def __init__(self, models, weights):
        self.models, self.weights = models, weights

    def predict(self, X):
        return sum(self.weights[k] * m.predict(X) for k, m in self.models.items())


def select_on_training(X, y, groups, spaces, outer_splits=5, inner_splits=3,
                       weights=DEFAULT_WEIGHTS, augment=None):
    """No test arguments. Nested CV avoids best-grid-score optimism in ranking."""
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite training data.')
    if y.ndim != 1 or len(X) != len(y) or len(groups) != len(y):
        raise ValueError('One target at a time, with aligned X/y/groups.')
    if pd.isna(groups).any() or len(np.unique(groups)) < 3:
        raise ValueError('At least three nonmissing groups needed for nested CV.')
    splits = list(GroupKFold(min(outer_splits, len(np.unique(groups)))).split(X, y, groups))
    fold_rows, cv_rows, models, params, oof = [], [], {}, {}, {}
    def search(space, xx, yy, gg):
        n = min(inner_splits, len(np.unique(gg)))
        if n < 2:
            raise ValueError('Not enough inner training groups.')
        model = FoldAugmentedRegressor(space['pipeline'], augment)
        grid = {'estimator__' + k: v for k, v in space['param_grid'].items()}
        return GridSearchCV(model, grid, scoring='neg_mean_absolute_error',
                            cv=GroupKFold(n), n_jobs=1, error_score='raise').fit(xx, yy, groups=gg)
    for name, space in spaces.items():
        pred = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(splits):
            if set(groups[tr]) & set(groups[va]) or len(va) < 2:
                raise ValueError('Invalid grouped split or fewer than two validation rows.')
            gs = search(space, X[tr], y[tr], groups[tr])
            pred[va] = gs.predict(X[va])
            fold_rows.append({'model': name, 'fold': fold, **metrics(y[va], pred[va])})
        oof[name] = pred
        rows = [r for r in fold_rows if r['model'] == name]
        cv_rows.append({'model': name, **{k: float(np.mean([r[k] for r in rows])) for k in CRITERIA}})
        gs = search(space, X, y, groups)
        models[name], params[name] = gs.best_estimator_, gs.best_params_
    table = pd.DataFrame(cv_rows)
    ranking = topsis(table, weights)
    baseline = table.sort_values(['MAE', 'model']).iloc[0]['model']
    winner = ranking.iloc[0]['model']
    ew = dict(zip(ranking.model, ranking.ensemble_weight))
    return {'ranking': ranking, 'fold_metrics': pd.DataFrame(fold_rows),
            'oof': pd.DataFrame({'actual': y, 'group': groups, **oof}),
            'baseline_name': baseline, 'topsis_name': winner, 'params': params,
            'models': models, 'ensemble': FrozenEnsemble(models, ew),
            'criteria_weights': list(np.asarray(weights) / np.sum(weights))}


def evaluate_frozen(selection, X_test, y_test, out_dir, audit, synthetic=False):
    """Test consumed only after selection. Does not alter ranking/weights."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    models = dict(selection['models'])
    models['CV_MAE_single'] = models[selection['baseline_name']]
    models['TOPSIS_single'] = models[selection['topsis_name']]
    models['TOPSIS_ensemble'] = selection['ensemble']
    predictions = pd.DataFrame({'actual': np.asarray(y_test)})
    rows = []
    for name, model in models.items():
        p = model.predict(np.asarray(X_test))
        predictions[name] = p
        rows.append({'model': name, **metrics(y_test, p)})
    report = pd.DataFrame(rows)
    for name, frame in [('ranking', selection['ranking']), ('fold_metrics', selection['fold_metrics']),
                        ('oof_predictions', selection['oof']), ('test_comparison', report),
                        ('test_predictions', predictions)]:
        frame.to_csv(out / (name + '.csv'), index=False)
    with pd.ExcelWriter(out / 'results.xlsx') as writer:
        selection['ranking'].to_excel(writer, sheet_name='CV_TOPSIS', index=False)
        report.to_excel(writer, sheet_name='frozen_test', index=False)
    record = {'synthetic': synthetic, 'isolation': audit, 'baseline': selection['baseline_name'],
              'topsis_single': selection['topsis_name'], 'criteria': CRITERIA,
              'directions': DIRECTIONS, 'weights': selection['criteria_weights'],
              'normalization': 'column L2', 'best_params': selection['params'],
              'test_used_for_selection': False}
    (out / 'audit.json').write_text(json.dumps(record, indent=2, default=str))
    joblib.dump(selection['ensemble'], out / 'TOPSIS_ensemble.pkl')
    joblib.dump(selection['models'][selection['baseline_name']], out / 'FINAL_CV_MAE.pkl')
    joblib.dump(selection['models'][selection['topsis_name']], out / 'FINAL_TOPSIS.pkl')
    rank = selection['ranking']
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(rank.model[::-1], rank.closeness[::-1])
    ax.set(xlabel='Training CV TOPSIS closeness', xlim=(0, 1), title='SIMULATED' if synthetic else 'TOPSIS')
    fig.tight_layout(); fig.savefig(out / 'topsis_ranking.png'); plt.close(fig)
    chosen = report[report.model.isin(['CV_MAE_single', 'TOPSIS_single', 'TOPSIS_ensemble'])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, CRITERIA):
        ax.bar(chosen.model, chosen[metric]); ax.set_title(metric); ax.tick_params(axis='x', rotation=35)
    fig.suptitle('SIMULATED held-out comparison' if synthetic else 'Frozen held-out comparison')
    fig.tight_layout(); fig.savefig(out / 'model_comparison.png'); plt.close(fig)
    return report


class LegacyAugment:
    """Reuse original SMOTER/Jitter/ResidualAug recipe inside each fit subset.

    cfg must be preregistered, not taken from the old test-optimized workbook.
    This single-target adapter changes the original three-target rare-tail OR rule.
    """
    def __init__(self, features, target, cfg):
        self.features, self.target, self.cfg = features, target, cfg

    def __call__(self, X, y):
        from run_topsis import legacy_namespace
        ns = legacy_namespace(1)
        cfg = self.cfg
        df = pd.DataFrame(X, columns=self.features); df[self.target] = y
        df = ns['ensure_base_tags'](df)
        rng = np.random.default_rng(42)
        df = ns['smoter'](df, [self.target], self.features, [],
                         rare_ratio=cfg['rare_ratio'],
                         q_per_target={self.target: tuple(cfg['quantiles'])}, rng=rng)
        df = ns['jitter'](df, ns['NOISE_SETTINGS'], ns['FEATURE_BOUNDS'],
                         n_samples=cfg['n_jitter'], seed=42)
        base = ns['XGBRegressor'](objective='reg:squarederror', random_state=42,
                                 n_jobs=1, **cfg['base_xgb'])
        df = ns['residual_augmentation'](df, [self.target], self.features, [], base,
                                        n_samples=cfg['n_residual'],
                                        beta_range=tuple(cfg['beta_range']), rng=rng)
        return df[self.features].to_numpy(float), df[self.target].to_numpy(float)
