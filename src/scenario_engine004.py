from itertools import product

import pandas as pd
import numpy as np

from src.features import compute_risk, add_derived_features

def get_product_baseline(d, product, col):
    rows = d[(d['product_name'] == product) & (d[col] > 0)]
    total_months = d[d['product_name'] == product]['month_idx'].nunique()
    n_sourced = len(rows)
    if n_sourced == 0:
        return None, total_months, 0
    return rows[col].mean(), total_months, n_sourced

def build_scenario_baseline(d, product, template_month_idx=10):
    row = d[(d['product_name'] == product) & (d['month_idx'] == template_month_idx)]
    if len(row) > 0:
        return row.copy(), template_month_idx
    # fallback: most recent month this product actually has
    product_rows = d[d['product_name'] == product]
    if len(product_rows) == 0:
        return None, None
    latest = product_rows.sort_values('month_idx').iloc[[-1]]
    return latest.copy(), latest['month_idx'].values[0]

def run_product_scenario(d, product, changes, price_booster, feature_cols,
                          cat_cols, product_vol_stats, scaling_ref, src_cols, imp_cols):
    baseline_result = build_scenario_baseline(d, product)
    if baseline_result[0] is None:
        return {'product_name': product, 'status': 'no_data_at_all'}

    baseline_row, baseline_month = baseline_result
    baseline_note = f"month {baseline_month}" + ("" if baseline_month == 10 else " (fallback, no month-10 data)")

    baseline_risk_row = compute_risk(baseline_row.copy(), product_vol_stats, scaling_ref)

    applicable = {}
    coverage_notes = []
    confidence_notes = []
    for col, factor in changes.items():
        base_val, total_months, n_sourced = get_product_baseline(d, product, col)
        if base_val is None:
            coverage_notes.append(f"{col}: never sourced ({total_months} months present)")
            continue

        applicable[col] = (base_val, factor)
        coverage_notes.append(f"{col}: sourced in {n_sourced}/{total_months} months")
        confidence_notes.append(f"{col}: {confidence_tag(n_sourced)}")

    if not applicable:
        return {'product_name': product, 
                'status': 'no_applicable_shock',
                'baseline_month': baseline_note,
                'coverage': '; '.join(coverage_notes),
                'confidence': '; '.join(confidence_notes)}

    shocked_row = baseline_row.copy()
    for col, (base_val, factor) in applicable.items():
        shocked_row[col] = base_val * factor

    shocked_row = add_derived_features(shocked_row, src_cols, imp_cols)
    X = shocked_row[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype('category')

    pred_price = np.expm1(price_booster.predict(X))[0]
    pred_volume = shocked_row['total_sources'].values[0]
    risk_row = compute_risk(shocked_row, product_vol_stats, scaling_ref)
    X_base = baseline_row[feature_cols].copy()

    # NEW — model's own baseline prediction, for a fair shock comparison
    X_base = baseline_row[feature_cols].copy()
    for c in cat_cols:
        X_base[c] = X_base[c].astype('category')
    pred_baseline = np.expm1(price_booster.predict(X_base))[0]

    return {
        'product_name': product,
        'status': 'ok',
        'baseline_month': baseline_note,
        'coverage': '; '.join(coverage_notes),
        'confidence': '; '.join(confidence_notes),
        'baseline_price': baseline_row['avg_price'].values[0],
        'model_baseline_price': pred_baseline,        # ADD THIS
        'predicted_price': pred_price,
        'price_delta_pct': (pred_price / pred_baseline - 1) * 100,
        'predicted_volume': pred_volume,
        'baseline_risk': baseline_risk_row['risk'].values[0],
        'baseline_risk_score': baseline_risk_row['risk_score'].values[0],
        'risk_score': risk_row['risk_score'].values[0],
        'risk': risk_row['risk'].values[0],
    }

def run_full_scenario(d, changes, **kwargs):
    results = []
    for p in d['product_name'].unique():
        r = run_product_scenario(d, p, changes, **kwargs)
        if r is not None:
            results.append(r)
    return pd.DataFrame(results)

def aggregate_summary(results_df):
    ok = results_df[results_df['status'] == 'ok']
    risk_order = {'Low': 0, 'Medium': 1, 'High': 2}
    risk_increased = (ok['risk'].map(risk_order) > ok['baseline_risk'].map(risk_order)).sum()
    return {
        'n_products_affected': len(ok),
        'n_products_na': (results_df['status'] != 'ok').sum(),
        'mean_price_delta_pct': ok['price_delta_pct'].mean(),
        'median_price_delta_pct': ok['price_delta_pct'].median(),
        'n_products_risk_increased': risk_increased,
    }

def confidence_tag(n_sourced):
    if n_sourced >= 4:
        return 'reliable'
    elif n_sourced >= 2:
        return 'low-confidence'
    else:
        return 'single-observation'

# Scenario runner
def run_scenario_report(d, changes, products=None, price_booster=None, feature_cols=None,
                          cat_cols=None, product_vol_stats=None, scaling_ref=None,
                          src_cols=None, imp_cols=None):
    """
    changes: dict, e.g. {'india': 0.7} or {'local': 1.2} or {'india': 0.7, 'local': 1.2}
    products: list of product names, or None to run on ALL products in d
    """
    target_products = products if products is not None else d['product_name'].unique()

    rows = []
    for p in target_products:
        result = run_product_scenario(d, p, changes, price_booster, feature_cols,
                                        cat_cols, product_vol_stats, scaling_ref, src_cols, imp_cols)
        rows.append(result)

    results_df = pd.DataFrame(rows)
    summary = aggregate_summary(results_df)
    return results_df, summary