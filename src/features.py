import pandas as pd
import numpy as np

def load_clean_data(path='data/processed/market_data_clean_v2.parquet'):
    d = pd.read_parquet(path)
    if 'naew' in d.columns:
        d = d.drop(columns=['naew'])   # still present in v2, confirmed — drop here
    d.columns = [c.lower() for c in d.columns]
    d = d.sort_values(['product_name', 'month_idx']).reset_index(drop=True)
    return d

exclude = ['product_name','category','bs_year','bs_month','month_idx','month_name',
           'volume','min_price','max_price','avg_price','unit','unit_canonical',
           'unit_changed','total_amount','volume_equals','total_sources',
           'reconciliation_gap','import_share','domestic_share','n_months_present','is_balanced',
           'india_share','china_share','bhutan_share','n_sources','herfindahl',
           'm_sin','m_cos','avg_price_lag1','volume_lag1']

cat_cols = ['product_name', 'category', 'unit']

def get_src_cols(d):
    return [c for c in d.columns if c not in exclude]


def compute_risk(df, product_vol_stats, scaling_ref):
    df = df.copy()
    df['volatility'] = (df['max_price'] - df['min_price']) / df['avg_price']
    df['concentration'] = 1 / df['n_sources'].replace(0, np.nan)
    df = df.merge(product_vol_stats, on='product_name', how='left')

    def scale01_fixed(x, lo, hi):
        return (x - lo) / (hi - lo + 1e-9)

    signal_cols = ['volatility', 'import_share', 'concentration', 'volume_cv']
    scaled = pd.DataFrame({
        c: scale01_fixed(df[c], *scaling_ref[c]) for c in signal_cols
    })
    df['risk_score'] = scaled.mean(axis=1, skipna=True)
    df['risk'] = pd.cut(df['risk_score'], bins=[-np.inf, 1/3, 2/3, np.inf], labels=['Low','Medium','High'])
    return df


## add this to features.py — used ONLY inside rebuild(), never in the normal load path
def add_derived_features(df, src_cols, imp_cols):
    """ Use this function ONLY INSIDE rebuild() - never in the normal load path. 
    It is used to recompute derived features on shocked data, which may have different 
    source values than the original data. """
    df = df.copy()

    share_cols = [c + '_share' for c in src_cols]
    stale = [c for c in share_cols + ['import_share','domestic_share','n_sources',
              'herfindahl','m_sin','m_cos','total_sources'] if c in df.columns]
    df = df.drop(columns=stale)   # remove the v2-baked-in versions before recomputing

    df['total_sources'] = df[src_cols].sum(axis=1)
    total_safe = df['total_sources'].replace(0, np.nan)

    shares = df[src_cols].div(total_safe, axis=0).fillna(0)
    shares.columns = share_cols
    df = pd.concat([df, shares], axis=1)

    df['import_share'] = df[imp_cols].sum(axis=1) / total_safe
    df['domestic_share'] = 1 - df['import_share'].fillna(0)
    df['import_share'] = df['import_share'].fillna(0)

    df['n_sources'] = (df[src_cols] > 0).sum(axis=1)
    df['herfindahl'] = (shares ** 2).sum(axis=1)

    df['m_sin'] = np.sin(2 * np.pi * df['bs_month'] / 12)
    df['m_cos'] = np.cos(2 * np.pi * df['bs_month'] / 12)
    return df

# def add_derived_features(df, src_cols, imp_cols):
#     df = df.copy()

#     # drop any pre-existing derived columns first — makes this safe to call
#     # on already-derived data (like v2) without creating duplicates
#     share_cols = [c + '_share' for c in src_cols]
#     stale = [c for c in share_cols + ['import_share','domestic_share','n_sources',
#               'herfindahl','m_sin','m_cos','total_sources'] if c in df.columns]
#     df = df.drop(columns=stale)

#     df['total_sources'] = df[src_cols].sum(axis=1)
#     total_safe = df['total_sources'].replace(0, np.nan)

#     shares = df[src_cols].div(total_safe, axis=0).fillna(0)
#     shares.columns = share_cols
#     df = pd.concat([df, shares], axis=1)

#     df['import_share'] = df[imp_cols].sum(axis=1) / total_safe
#     df['domestic_share'] = 1 - df['import_share'].fillna(0)
#     df['import_share'] = df['import_share'].fillna(0)

#     df['n_sources'] = (df[src_cols] > 0).sum(axis=1)
#     df['herfindahl'] = (shares ** 2).sum(axis=1)

#     df['m_sin'] = np.sin(2 * np.pi * df['bs_month'] / 12)
#     df['m_cos'] = np.cos(2 * np.pi * df['bs_month'] / 12)
#     return df

# def add_lags(df, src_cols, imp_cols):
#     df = df.sort_values(['product_name', 'month_idx']).copy()
#     df = df.drop(columns=[c for c in ['avg_price_lag1','volume_lag1'] if c in df.columns])
#     df = add_derived_features(df, src_cols, imp_cols)
#     g = df.groupby('product_name')
#     df['avg_price_lag1'] = g['avg_price'].shift(1)
#     df['volume_lag1']    = g['total_sources'].shift(1)
#     return df

# def get_feature_cols(src_cols):
#     cols = (
#         ['product_name', 'category', 'unit', 'm_sin', 'm_cos', 'n_sources',
#          'herfindahl', 'import_share', 'domestic_share', 'n_months_present',
#          'india_share', 'china_share', 'bhutan_share', 'avg_price_lag1']
#         + src_cols
#     )
#     assert len(cols) == len(set(cols)), "duplicate columns in feature_cols"
#     return cols
