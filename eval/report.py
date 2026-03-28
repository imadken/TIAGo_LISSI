#!/usr/bin/env python3
"""
report.py — Compile all evaluation JSON results into figures and a summary
            table ready for the paper.

Reads all data/*.json files produced by the evaluation suite and generates:
  • report_figures/fig_latency.pdf       — Component latency bar chart
  • report_figures/fig_detection.pdf     — YOLO detection rate + confidence per class
  • report_figures/fig_depth.pdf         — Depth refinement: area reduction + IoU
  • report_figures/fig_grasp.pdf         — Grasp success rate per class
  • report_figures/fig_search.pdf        — Head search: positions-to-find histogram
  • report_figures/fig_statemanager.pdf  — Spatial relation accuracy per type
  • report_figures/summary_table.csv     — Single CSV summarising all metrics

Usage
─────
    cd /home/lissi/tiago_public_ws/eval
    python3 report.py
    python3 report.py --out_dir my_figs   # custom output directory
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import load_result

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    'blue':   '#2B7BB9',
    'orange': '#E87C2B',
    'green':  '#3BAA6A',
    'red':    '#C94040',
    'purple': '#7B4FA6',
    'grey':   '#8A8A8A',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.35,
    'figure.dpi': 150,
})


def savefig(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


# ── Latency figure ─────────────────────────────────────────────────────────────

def fig_latency(data: list, out_dir: str):
    # Find the most recent full-run record (has at least YOLO and depth)
    records = [d for d in data if isinstance(d, dict) and 'component' in d]
    if not records:
        print('[skip] latency: no component records')
        return

    comp_map = {r['component']: r for r in records if r.get('n', 0) > 0}
    targets = ['YOLO inference', 'VLM API (Gemini)',
               'Depth refinement', 'RANSAC cylinder', 'Face recognition']
    present = [c for c in targets if c in comp_map]
    if not present:
        print('[skip] latency: no usable components')
        return

    means  = [comp_map[c]['mean_s'] * 1000  for c in present]
    p95s   = [comp_map[c]['p95_s']  * 1000  for c in present]
    p50s   = [comp_map[c]['p50_s']  * 1000  for c in present]
    stds   = [comp_map[c]['std_s']  * 1000  for c in present]

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(x, means, width=0.5, color=COLORS['blue'], alpha=0.85,
                  label='Mean', zorder=3)
    ax.scatter(x, p50s, marker='D', color=COLORS['green'], s=30, zorder=5,
               label='p50')
    ax.scatter(x, p95s, marker='^', color=COLORS['red'],   s=30, zorder=5,
               label='p95')
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#333', elinewidth=1,
                capsize=4, zorder=6)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    short_labels = [c.replace(' inference', '').replace(' (Gemini)', '')
                    .replace(' refinement', '\nrefinement')
                    .replace(' recognition', '\nrecognition')
                    for c in present]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Component Latency Profile')
    ax.legend(loc='upper right', framealpha=0.8)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_latency.pdf'))


# ── Detection figure ───────────────────────────────────────────────────────────

def fig_detection(data: list, out_dir: str):
    if not data:
        print('[skip] detection: empty')
        return
    classes = sorted({r.get('class', '') for r in data if r.get('class')})
    det_rates = []
    conf_means = []
    for cls in classes:
        rows = [r for r in data if r.get('class') == cls]
        det_rates.append(np.mean([r.get('yolo_detected', False) for r in rows]) * 100)
        confs = [r['yolo_conf'] for r in rows if r.get('yolo_conf')]
        conf_means.append(np.mean(confs) if confs else 0.0)

    x = np.arange(len(classes))
    fig, ax1 = plt.subplots(figsize=(6.0, 3.5))
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)

    b1 = ax1.bar(x - 0.18, det_rates, width=0.35, color=COLORS['blue'],
                 alpha=0.85, label='Detection rate (%)', zorder=3)
    b2 = ax2.bar(x + 0.18, conf_means, width=0.35, color=COLORS['orange'],
                 alpha=0.85, label='Mean confidence', zorder=3)

    ax1.set_ylabel('Detection Rate (%)')
    ax2.set_ylabel('YOLO Confidence')
    ax1.set_ylim(0, 115)
    ax2.set_ylim(0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_title('YOLO Detection Performance by Object Class')
    lines = [mpatches.Patch(color=COLORS['blue'],   label='Detection rate (%)'),
             mpatches.Patch(color=COLORS['orange'], label='Mean confidence')]
    ax1.legend(handles=lines, loc='upper right', framealpha=0.8)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_detection.pdf'))


# ── Depth refinement figure ────────────────────────────────────────────────────

def fig_depth(data: list, out_dir: str):
    if not data:
        print('[skip] depth: empty')
        return
    # Flatten any nested structure (list of scene-level dicts with 'raw' list)
    rows = []
    for entry in data:
        if isinstance(entry, list):
            rows.extend(entry)
        elif isinstance(entry, dict) and 'raw' in entry:
            rows.extend(entry['raw'])
        elif isinstance(entry, dict) and 'area_reduction' in entry:
            rows.append(entry)

    if not rows:
        print('[skip] depth: no row-level data')
        return

    area_reds = [r['area_reduction'] for r in rows if r.get('area_reduction') is not None]
    iou_vlm   = [r['iou_vlm_yolo']       for r in rows if r.get('iou_vlm_yolo') is not None]
    iou_ref   = [r['iou_refined_yolo']   for r in rows if r.get('iou_refined_yolo') is not None]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Panel A: area reduction distribution
    ax = axes[0]
    if area_reds:
        ax.hist(area_reds, bins=15, color=COLORS['blue'], alpha=0.8, edgecolor='white')
        ax.axvline(np.mean(area_reds), color=COLORS['red'], lw=1.5,
                   label=f'Mean={np.mean(area_reds):.2f}')
        ax.set_xlabel('Bbox Area Reduction (fraction)')
        ax.set_ylabel('Count')
        ax.set_title('Depth Bbox Refinement\n(positive = tighter)')
        ax.legend()

    # Panel B: IoU before/after refinement
    ax = axes[1]
    both = [(a, b) for a, b in zip(iou_vlm, iou_ref)]
    if both:
        iou_v, iou_r = zip(*both)
        x = np.arange(min(len(iou_v), 20))  # show up to 20 samples
        ax.plot(x, iou_v[:20], 'o--', color=COLORS['orange'], label='VLM bbox IoU',  ms=4)
        ax.plot(x, iou_r[:20], 's-',  color=COLORS['green'],  label='Refined bbox IoU', ms=4)
        ax.set_xlabel('Sample index')
        ax.set_ylabel('IoU vs YOLO ground truth')
        ax.set_title('Bbox IoU Improvement\nAfter Depth Refinement')
        ax.set_ylim(0, 1.05)
        ax.legend()

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_depth.pdf'))


# ── Grasp figure ───────────────────────────────────────────────────────────────

def fig_grasp(data: list, out_dir: str):
    if not data:
        print('[skip] grasp: empty')
        return
    classes = sorted({r.get('object_class', '') for r in data if r.get('object_class')})
    success  = [np.mean([r.get('grasp_success', False)
                         for r in data if r.get('object_class') == c]) * 100
                for c in classes]
    det_rate = [np.mean([r.get('pre_detection', False)
                         for r in data if r.get('object_class') == c]) * 100
                for c in classes]
    durations = [np.mean([r.get('duration_s', 0)
                          for r in data if r.get('object_class') == c])
                 for c in classes]

    x = np.arange(len(classes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Success + detection rate
    b1 = ax1.bar(x - 0.18, success,  width=0.35, color=COLORS['green'],  alpha=0.85,
                 label='Grasp success (%)', zorder=3)
    b2 = ax1.bar(x + 0.18, det_rate, width=0.35, color=COLORS['blue'],   alpha=0.85,
                 label='Pre-grasp detection (%)', zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels(classes)
    ax1.set_ylabel('%')
    ax1.set_ylim(0, 115)
    ax1.set_title('Grasp Success & Detection Rate')
    ax1.legend(fontsize=7)
    for b, v in zip(b1, success):
        ax1.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}%', ha='center', fontsize=7)

    # Attempt duration
    ax2.bar(x, durations, width=0.5, color=COLORS['orange'], alpha=0.85, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(classes)
    ax2.set_ylabel('Seconds')
    ax2.set_title('Mean Grasp Attempt Duration')
    for i, (xi, d) in enumerate(zip(x, durations)):
        ax2.text(xi, d+0.5, f'{d:.1f}s', ha='center', fontsize=7)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_grasp.pdf'))


# ── Search figure ──────────────────────────────────────────────────────────────

def fig_search(data: list, out_dir: str):
    if not data:
        print('[skip] search: empty')
        return
    classes = sorted({r.get('target_class', '') for r in data if r.get('target_class')})

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # Panel A: histogram of positions_scanned for found trials
    ax = axes[0]
    found = [r for r in data if r.get('target_found')]
    if found:
        pos_counts = [r['positions_scanned'] for r in found]
        bins = np.arange(0.5, 13.5, 1)
        ax.hist(pos_counts, bins=bins, color=COLORS['blue'], alpha=0.85, edgecolor='white')
        ax.axvline(np.mean(pos_counts), color=COLORS['red'], lw=1.5,
                   label=f'Mean={np.mean(pos_counts):.1f}')
        ax.set_xlabel('Positions scanned before detection')
        ax.set_ylabel('Trials')
        ax.set_title('Head Search Efficiency\n(positions to find target)')
        ax.set_xticks(range(1, 13))
        ax.legend()

    # Panel B: success rate and mean time per class
    ax = axes[1]
    sr    = []
    t_avg = []
    for cls in classes:
        rows = [r for r in data if r.get('target_class') == cls]
        sr.append(np.mean([r.get('target_found', False) for r in rows]) * 100)
        t_avg.append(np.mean([r.get('search_duration_s', 0) for r in rows]))

    x = np.arange(len(classes))
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax.bar(x - 0.18, sr,    width=0.35, color=COLORS['green'],  alpha=0.85, label='Success %')
    ax2.bar(x + 0.18, t_avg, width=0.35, color=COLORS['orange'], alpha=0.85, label='Mean time (s)')
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('Success Rate (%)')
    ax2.set_ylabel('Mean Search Time (s)')
    ax.set_title('Search Success & Duration\nby Target Class')
    h1 = mpatches.Patch(color=COLORS['green'],  label='Success %')
    h2 = mpatches.Patch(color=COLORS['orange'], label='Mean time (s)')
    ax.legend(handles=[h1, h2], loc='lower right', fontsize=7)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_search.pdf'))


# ── State manager figure ───────────────────────────────────────────────────────

def fig_statemanager(data: list, out_dir: str):
    if not data:
        print('[skip] statemanager: empty')
        return
    # last entry that has per_relation
    entry = None
    for d in reversed(data):
        if isinstance(d, dict) and 'per_relation' in d:
            entry = d
            break
    if entry is None:
        print('[skip] statemanager: no per_relation data')
        return

    pr = entry['per_relation']
    relations = list(pr.keys())
    accs = [pr[r]['acc'] * 100 for r in relations]
    ns   = [pr[r]['n']         for r in relations]

    x = np.arange(len(relations))
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(x, accs, width=0.5,
                  color=[COLORS['green'] if a >= 80 else COLORS['orange']
                         if a >= 60 else COLORS['red'] for a in accs],
                  alpha=0.85, zorder=3)
    ax.axhline(100, color='#aaa', lw=0.8, ls='--')
    ax.axhline(80,  color=COLORS['blue'], lw=1.0, ls=':', label='80% threshold')

    for bar, acc, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{acc:.0f}%\n(n={n})', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(relations, rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 120)
    ax.set_title('Scene-Graph Spatial Relation Accuracy')
    ax.legend()
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'fig_statemanager.pdf'))


# ── Summary CSV ────────────────────────────────────────────────────────────────

def write_summary_csv(results: dict, out_dir: str):
    rows = []

    # Latency
    for r in results.get('latency', []):
        if isinstance(r, dict) and 'component' in r and r.get('n', 0) > 0:
            rows.append({
                'metric_group': 'Latency',
                'metric': r['component'],
                'value': f"{r.get('mean_s', 0)*1000:.1f} ms",
                'detail': f"p50={r.get('p50_s',0)*1000:.0f}ms  p95={r.get('p95_s',0)*1000:.0f}ms  std={r.get('std_s',0)*1000:.0f}ms",
            })
        elif isinstance(r, dict) and r.get('component') == 'e2e_budget_s':
            rows.append({
                'metric_group': 'Latency',
                'metric': 'End-to-end budget (model)',
                'value': f"{r.get('total', '?')} s",
                'detail': str(r.get('breakdown', '')),
            })

    # Detection
    det_data = results.get('detection', [])
    det_classes = sorted({r.get('class','') for r in det_data if r.get('class')})
    for cls in det_classes:
        rows_cls = [r for r in det_data if r.get('class') == cls]
        det = np.mean([r.get('yolo_detected', False) for r in rows_cls]) * 100
        confs = [r['yolo_conf'] for r in rows_cls if r.get('yolo_conf')]
        rows.append({
            'metric_group': 'Detection',
            'metric': f'{cls} detection rate',
            'value': f'{det:.1f}%',
            'detail': f'mean_conf={np.mean(confs):.3f}  n={len(rows_cls)}' if confs else f'n={len(rows_cls)}',
        })

    # Grasp
    grasp_data = results.get('grasp_trials', [])
    grasp_classes = sorted({r.get('object_class','') for r in grasp_data if r.get('object_class')})
    for cls in grasp_classes:
        rows_cls = [r for r in grasp_data if r.get('object_class') == cls]
        sr = np.mean([r.get('grasp_success', False) for r in rows_cls]) * 100
        dur = np.mean([r.get('duration_s', 0) for r in rows_cls])
        rows.append({
            'metric_group': 'Grasp',
            'metric': f'{cls} grasp success',
            'value': f'{sr:.1f}%',
            'detail': f'mean_duration={dur:.1f}s  n={len(rows_cls)}',
        })

    # Search
    search_data = results.get('search_trials', [])
    if search_data:
        found = [r for r in search_data if r.get('target_found')]
        sr  = len(found) / len(search_data) * 100
        pos = np.mean([r['positions_scanned'] for r in found]) if found else float('nan')
        dur = np.mean([r.get('search_duration_s', 0) for r in search_data])
        rows.append({
            'metric_group': 'Search',
            'metric': 'Head search success rate',
            'value': f'{sr:.1f}%',
            'detail': f'mean_positions={pos:.1f}  mean_duration={dur:.1f}s  n={len(search_data)}',
        })

    # Face
    face_data = results.get('face_recognition', [])
    for entry in face_data:
        if isinstance(entry, dict) and 'accuracy' in entry:
            rows.append({
                'metric_group': 'Face Recognition',
                'metric': 'Accuracy',
                'value': f"{entry['accuracy']*100:.1f}%",
                'detail': f"FAR={entry.get('far',0)*100:.1f}%  FRR={entry.get('frr',0)*100:.1f}%  n={entry.get('n_test','?')}",
            })

    # Depth
    depth_data = results.get('depth_refinement', [])

    # State manager
    sm_data = results.get('statemanager', [])
    for entry in sm_data:
        if isinstance(entry, dict) and 'accuracy' in entry:
            rows.append({
                'metric_group': 'State Manager',
                'metric': 'Spatial relation accuracy',
                'value': f"{entry['accuracy']*100:.1f}%",
                'detail': f"n_relations={entry.get('n_relations','?')}  n_scenes={entry.get('n_scenes','?')}",
            })
            for rel, stat in entry.get('per_relation', {}).items():
                rows.append({
                    'metric_group': 'State Manager',
                    'metric': f'  {rel}',
                    'value': f"{stat['acc']*100:.1f}%",
                    'detail': f"n={stat['n']}",
                })

    path = os.path.join(out_dir, 'summary_table.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric_group', 'metric', 'value', 'detail'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'  saved → {path}  ({len(rows)} rows)')


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    dataset_names = [
        'latency', 'detection', 'depth_refinement',
        'grasp_trials', 'search_trials', 'face_recognition', 'statemanager',
    ]
    results = {name: load_result(name) for name in dataset_names}

    found = [n for n, d in results.items() if d]
    missing = [n for n, d in results.items() if not d]
    print(f'\nLoaded: {found}')
    if missing:
        print(f'Missing (will skip): {missing}')

    print(f'\nGenerating figures → {out_dir}/')
    fig_latency(results['latency'],              out_dir)
    fig_detection(results['detection'],          out_dir)
    fig_depth(results['depth_refinement'],       out_dir)
    fig_grasp(results['grasp_trials'],           out_dir)
    fig_search(results['search_trials'],         out_dir)
    fig_statemanager(results['statemanager'],    out_dir)

    print(f'\nWriting summary CSV …')
    write_summary_csv(results, out_dir)

    print('\n[done] report complete.')
    print(f'  Figures and CSV are in: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='report_figures',
                    help='output directory for figures and CSV')
    args = ap.parse_args()
    main(args)
