from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/src')
from src.features import load_clean_data, get_src_cols, cat_cols
from src.scenario_engine004 import run_scenario_report

app = Flask(__name__)

# Only lightweight globals
QUICK_SOURCES = ['india', 'local', 'china', 'kathmandu', 'bhutan', 'birgunj']

def make_charts(results_df):
    """Returns two base64-encoded PNGs: sensitivity bar chart + risk shift chart."""
    ok = results_df[results_df['status'] == 'ok'].copy()
    ok = ok.sort_values('price_delta_pct')

    # chart 1: top 15 most-affected products, price delta
    top = ok.reindex(ok['price_delta_pct'].abs().sort_values(ascending=False).index).head(15)
    top = top.sort_values('price_delta_pct')
    colors = ['#c0392b' if v > 0 else '#2c7fb8' for v in top['price_delta_pct']]

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.barh(top['product_name'], top['price_delta_pct'], color=colors)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_xlabel('Price change (%)')
    ax1.set_title('Top 15 most affected products')
    plt.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=110)
    plt.close(fig1)
    chart1 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    # chart 2: risk category counts, baseline vs shocked
    risk_order = ['Low', 'Medium', 'High']
    baseline_counts = ok['baseline_risk'].value_counts().reindex(risk_order, fill_value=0)
    shocked_counts = ok['risk'].value_counts().reindex(risk_order, fill_value=0)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    x = np.arange(len(risk_order))
    width = 0.35
    ax2.bar(x - width/2, baseline_counts.values, width, label='Before', color='#999999')
    ax2.bar(x + width/2, shocked_counts.values, width, label='After', color='#c0392b')
    ax2.set_xticks(x)
    ax2.set_xticklabels(risk_order)
    ax2.set_ylabel('Number of products')
    ax2.set_title('Risk category shift')
    ax2.legend()
    plt.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=110)
    plt.close(fig2)
    chart2 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    # (same chart code as before)
    return chart1, chart2


@app.route('/', methods=['GET', 'POST'])
def index():
    results_table = None
    summary = None
    selected_sources = []
    factors = {}
    error = None
    chart1 = chart2 = None
    example_up = None
    example_down = None
    n_price_up = None
    n_price_down = None

    # 🔑 Lazy load heavy data here
    d = load_clean_data()
    src_cols = get_src_cols(d)
    imp_cols = [c for c in ['india', 'china', 'bhutan'] if c in src_cols]
    feature_cols = (
        ['product_name','category','unit','m_sin','m_cos','n_sources',
         'herfindahl','import_share','domestic_share','n_months_present',
         'india_share','china_share','bhutan_share','avg_price_lag1']
        + src_cols
    )
    booster = lgb.Booster(model_file='models/price_surrogate_final.txt')
    product_vol_stats = pd.read_parquet('data/processed/product_volume_stats.parquet').reset_index()
    with open('data/processed/risk_scaling_ref.json') as f:
        scaling_ref = json.load(f)

    SOURCE_OPTIONS = sorted(src_cols)
    QUICK = [s for s in QUICK_SOURCES if s in src_cols]
    MORE = sorted([s for s in src_cols if s not in QUICK])

    if request.method == 'POST':
        selected_sources = request.form.getlist('sources')
        changes = {}
        for src in selected_sources:
            pct = request.form.get(f'pct_{src}', '')
            try:
                factor = 1 + (float(pct) / 100)
                changes[src] = factor
                factors[src] = pct
            except ValueError:
                continue

        if not changes:
            error = "Pick at least one source and enter a valid % change."
        else:
            results_df, summary = run_scenario_report(
                d, changes, products=None,
                price_booster=booster, feature_cols=feature_cols, cat_cols=cat_cols,
                product_vol_stats=product_vol_stats, scaling_ref=scaling_ref,
                src_cols=src_cols, imp_cols=imp_cols
            )
            ok = results_df[results_df['status'] == 'ok'].sort_values('price_delta_pct', ascending=False)
            results_table = ok.to_dict('records')
            chart1, chart2 = make_charts(results_df)

            ok_for_counts = results_df[results_df['status'] == 'ok']
            n_price_up = int((ok_for_counts['price_delta_pct'] > 0).sum())
            n_price_down = int((ok_for_counts['price_delta_pct'] < 0).sum())

            example_up = ok[ok['price_delta_pct'] > 0].sort_values('price_delta_pct', ascending=False)
            example_down = ok[ok['price_delta_pct'] < 0].sort_values('price_delta_pct')
            example_up = example_up.iloc[0].to_dict() if len(example_up) > 0 else None
            example_down = example_down.iloc[0].to_dict() if len(example_down) > 0 else None

    return render_template(
        'index.html',
        source_options=SOURCE_OPTIONS,
        selected_sources=selected_sources,
        factors=factors,
        results_table=results_table,
        summary=summary,
        error=error,
        chart1=chart1,
        chart2=chart2,
        quick_sources=QUICK,
        more_sources=MORE,
        example_up=example_up,
        example_down=example_down,
        n_price_up=n_price_up,
        n_price_down=n_price_down,
    )

if __name__ == '__main__':
    app.run(debug=True)
