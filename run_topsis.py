"""Safe default: simulated smoke run. Real training requires --real explicitly."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from topsis_ensemble import assert_isolation, select_on_training, evaluate_frozen, LegacyAugment


def legacy_namespace(cell):
    """Load definitions/configuration, never execute Notebook main/Optuna jobs."""
    notebook = json.loads(Path(__file__).with_name('shuju.ipynb').read_text())
    tree = ast.parse(''.join(notebook['cells'][cell]['source']))
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom,
                  ast.FunctionDef, ast.Assign, ast.AnnAssign)) or
                  (isinstance(n, ast.Try) and cell == 7)]
    ns = {'__name__': 'legacy_definitions'}
    exec(compile(tree, '<legacy definitions>', 'exec'), ns)
    return ns


def jitter(X, y):
    rng = np.random.default_rng(42)
    return np.vstack([X, X + rng.normal(size=X.shape) * np.std(X, axis=0) * .01]), np.tile(y, 2)


def run(args):
    if args.real:
        if not args.data:
            raise ValueError('--real requires --data raw.xlsx; never use train_augmented_best.')
        ns = legacy_namespace(7)
        features = ns['FEATURES']
        df = pd.read_excel(args.data, sheet_name=args.sheet)
        if 'origin' in df and not df.origin.astype(str).str.lower().eq('original').all():
            raise ValueError('Only unaugmented original rows allowed.')
        if 'row_id' not in df:
            df['row_id'] = np.arange(len(df))
        if df.row_id.isna().any() or df.row_id.duplicated().any():
            raise ValueError('Original row IDs must be complete and unique.')
        # Parent-connected components AND supplied scientific source grouping.
        component = ns['add_group_id'](df)['group_id'].to_numpy()
        group_col = args.group_col
        if group_col not in df or df[group_col].isna().any():
            raise ValueError('Provide complete --group-col for experimental source/batch.')
        # Merge overlapping source and ancestry partitions, rather than overwrite either.
        parent = np.arange(len(df))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for values in [component, df[group_col].to_numpy()]:
            first = {}
            for i, v in enumerate(values):
                if v in first: parent[find(i)] = find(first[v])
                else: first[v] = i
        df['group_id'] = [find(i) for i in range(len(df))]
        spaces = ns['build_model_spaces']()
        if not args.full_grid:
            spaces = {k: v for k, v in spaces.items() if k in ['DT', 'EN']}
            for s in spaces.values():
                s['param_grid'] = {k: [v[0]] for k, v in s['param_grid'].items()}
        augment = jitter if args.augment == 'jitter' else None
    else:
        rng = np.random.default_rng(42)
        features = ['x1', 'x2', 'x3']
        df = pd.DataFrame(rng.normal(size=(90, 3)), columns=features)
        for i, target in enumerate(['Hemicellulose', 'Cellulose', 'Lignin']):
            df[target] = (i + 1) * df.x1 - df.x2 + rng.normal(0, .3, len(df))
        df['row_id'] = np.arange(len(df)); df['group_id'] = np.repeat(np.arange(30), 3)
        spaces = {
            'Ridge': {'pipeline': make_pipeline(StandardScaler(), Ridge()),
                      'param_grid': {'ridge__alpha': [.1, 1]}},
            'DT': {'pipeline': DecisionTreeRegressor(random_state=42),
                   'param_grid': {'max_depth': [2, 4]}},
            'EN': {'pipeline': make_pipeline(StandardScaler(), ElasticNet(max_iter=2000)),
                   'param_grid': {'elasticnet__alpha': [.01, .1]}}}
        augment = jitter
    targets = ['Hemicellulose', 'Cellulose', 'Lignin']
    if not np.isfinite(df[features + targets].to_numpy(float)).all():
        raise ValueError('Missing/nonfinite inputs; fix before splitting (no silent row removal).')
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(df, groups=df.group_id))
    train, test = df.iloc[tr], df.iloc[te]
    audit = assert_isolation(train, test, features)
    # Fit/select ALL targets before opening their test labels for evaluation.
    selections = {target: select_on_training(train[features], train[target], train.group_id,
                  spaces, outer_splits=args.outer, inner_splits=args.inner, augment=(LegacyAugment(features, target, json.loads(Path(args.augmentation_config).read_text()))
                           if args.augmentation_config else augment))
                  for target in targets}
    for target, selection in selections.items():
        evaluate_frozen(selection, test[features], test[target], Path(args.out) / target,
                        audit, synthetic=not args.real)
    Path(args.out, 'split_ids.json').write_text(json.dumps({'train': train.row_id.tolist(), 'test': test.row_id.tolist()}))
    print('Completed:', args.out, '| SIMULATED' if not args.real else '| REAL')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--real', action='store_true'); p.add_argument('--data')
    p.add_argument('--sheet', default='Sheet1'); p.add_argument('--group-col', default='source_id')
    p.add_argument('--full-grid', action='store_true')
    p.add_argument('--augmentation-config', help='Preregistered JSON; reuses original SMOTER/Jitter/ResidualAug')
    p.add_argument('--augment', choices=['none', 'jitter'], default='none')
    p.add_argument('--outer', type=int, default=3); p.add_argument('--inner', type=int, default=2)
    p.add_argument('--out', default='results_simulated')
    run(p.parse_args())
