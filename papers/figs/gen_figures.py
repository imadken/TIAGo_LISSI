#!/usr/bin/env python3
"""Generate all figures for the TIAGo paper. Run from papers/figs/."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

C_NAVY   = '#003E7F'
C_BLUE   = '#2563EB'
C_LBLUE  = '#DBEAFE'
C_GREEN  = '#16A34A'
C_LGREEN = '#DCFCE7'
C_AMBER  = '#D97706'
C_LAMBER = '#FEF3C7'
C_RED    = '#E11D48'
C_LRED   = '#FFE4E6'
C_PURPLE = '#7C3AED'
C_LPURP  = '#EDE9FE'
C_GRAY   = '#6B7280'
C_SLATE  = '#475569'
C_TEAL   = '#0F766E'
C_LTEAL  = '#CCFBF1'

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, line1, line2='', line3='',
        fc='#DBEAFE', ec='#2563EB', fs=9, bold=True, pad=0.1):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=f'round,pad={pad}',
                        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3)
    ax.add_patch(b)
    lines = [l for l in [line1, line2, line3] if l]
    n = len(lines)
    offsets = {1: [0], 2: [0.14, -0.14], 3: [0.25, 0, -0.25]}[n]
    for txt, dy in zip(lines, offsets):
        fw = 'bold' if bold and txt == line1 else 'normal'
        fs_use = fs if txt == line1 else fs - 1.0
        col = ec if txt == line1 else C_SLATE
        ax.text(x, y + dy * h, txt, ha='center', va='center',
                fontsize=fs_use, fontweight=fw, color=col, zorder=4)

def arr(ax, x0, y0, x1, y1, color=C_SLATE, label='', lw=1.5,
        rad=0.0, bidirectional=False):
    style = '<->' if bidirectional else '->'
    cs = f'arc3,rad={rad}' if rad else 'arc3,rad=0'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=cs), zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=7.2,
                color=color, zorder=6,
                bbox=dict(fc='white', ec='none', pad=1, alpha=0.85))

def cluster(ax, x0, y0, x1, y1, label, fc, ec):
    b = FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                        boxstyle='round,pad=0.05',
                        facecolor=fc, edgecolor=ec, linewidth=1.1,
                        linestyle='--', alpha=0.5, zorder=1)
    ax.add_patch(b)
    ax.text((x0+x1)/2, y1+0.06, label, ha='center', va='bottom',
            fontsize=8, color=ec, fontweight='bold', zorder=2)

# =============================================================================
# FIG 1 — System Architecture  (clean 4-row layout, 13×6.5 in)
# =============================================================================
fig1, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.5)
ax.axis('off')
fig1.patch.set_facecolor('white')

# ── Row 1: Speech pipeline ────────────────────────────────────────────────
cluster(ax, 0.2, 5.3, 12.8, 6.35, 'Speech Interface', '#EFF6FF', C_BLUE)
box(ax,  1.1, 5.8, 1.5, 0.7, 'Microphone', 'Voice Input',   fc=C_LBLUE, ec=C_BLUE)
box(ax,  3.0, 5.8, 1.6, 0.7, 'Speech-to-Text',   'Automatic Recognition',        fc=C_LBLUE, ec=C_BLUE)
box(ax,  6.5, 5.8, 2.4, 0.7, 'Vision-Language Model', 'Gemini 2.0 Flash',
    'plan  ·  detect  ·  verify',  fc=C_LPURP, ec=C_PURPLE, fs=9.5)
box(ax, 10.0, 5.8, 1.6, 0.7, 'Text-to-Speech',    'Verbal Response',             fc=C_LBLUE, ec=C_BLUE)
box(ax, 11.9, 5.8, 1.4, 0.7, 'Speaker','Audio Output',                 fc=C_LBLUE, ec=C_BLUE)

arr(ax, 1.85, 5.8, 2.2, 5.8, C_BLUE, 'audio')
arr(ax, 3.8,  5.8, 5.3, 5.8, C_BLUE, 'command text')
arr(ax, 7.7,  5.8, 9.2, 5.8, C_BLUE, 'response')
arr(ax,10.8,  5.8, 11.2,5.8, C_BLUE)

# ── Row 2: Perception + FaceManager ───────────────────────────────────────
cluster(ax, 0.2, 3.8, 5.8, 5.1, 'Perception', '#F0FDF4', C_GREEN)
cluster(ax, 6.0, 3.8, 12.8,5.1, 'State & Identity', '#FFFBEB', C_AMBER)

box(ax, 1.1,  4.45, 1.6, 0.8, 'RGBD Camera', 'Colour + Depth',    fc=C_LGREEN, ec=C_GREEN)
box(ax, 3.2,  4.45, 2.0, 0.8, 'Perception', 'Image Capture',
    'Depth Acquisition',               fc=C_LGREEN, ec=C_GREEN)
box(ax, 7.5,  4.45, 2.6, 0.8, 'Scene Memory',  'Spatial Relations',
    'Object & Robot State', fc=C_LAMBER, ec=C_AMBER, fs=9)
box(ax,11.1,  4.45, 2.8, 0.8, 'Face Recognition', 'Person Identification',
    'Identity Memory',     fc=C_LAMBER, ec=C_AMBER)

arr(ax, 1.9,  4.45, 2.2, 4.45, C_GREEN)
arr(ax, 4.2,  4.45, 5.7, 4.45, C_PURPLE, 'RGB image')
arr(ax, 4.2,  4.7,  6.2, 4.7,  C_AMBER,  'detections')

# RGBD → FaceManager
arr(ax, 2.0,  4.1,  9.7,  4.1, C_AMBER,  'RGB frames', rad=-0.15)
arr(ax, 9.8,  4.45, 9.8,  4.05, C_AMBER)   # down
arr(ax, 9.8,  4.05,10.7, 4.05, C_AMBER)   # right into FM left boundary (approx)

# FaceManager → StateManager
arr(ax, 9.7, 4.45, 9.2, 4.45, C_AMBER, 'identity')

# VLM ↔ StateManager
arr(ax, 6.5, 5.42, 8.0, 5.1,  C_PURPLE, 'detections', rad=0.0)
arr(ax, 7.5, 5.1,  6.5, 5.42, C_PURPLE, 'context',    rad=0.25)

# ── Row 3: Skill Library ──────────────────────────────────────────────────
cluster(ax, 0.2, 2.1, 12.8,3.65, 'Skill Library  (BaseSkill: check_affordance · execute · verify)', '#F0FDF4', C_GREEN)

box(ax, 1.5,  2.85, 2.2, 0.85, 'Visual Search',     'Head scanning',
    '12 gaze positions',          fc=C_LGREEN, ec=C_GREEN)
box(ax, 4.3,  2.85, 2.2, 0.85, 'Grasp Object',      'RANSAC 3D pose',
    'Arm + torso motion',         fc=C_LGREEN, ec=C_GREEN)
box(ax, 7.1,  2.85, 2.2, 0.85, 'Handover',          'Approach person',
    'Extend arm, release',        fc=C_LGREEN, ec=C_GREEN)
box(ax,10.3,  2.85, 3.6, 0.85, 'Motion Primitives',
    'Go home  ·  Wave  ·  Open / Close hand',
    '',                           fc='#F1F5F9', ec=C_SLATE, bold=False)

# VLM → skills
arr(ax, 6.5, 5.42, 1.5, 3.28, C_GREEN, 'skill select', rad=0.1)
arr(ax, 6.5, 5.42, 4.3, 3.28, C_GREEN, '', rad=0.05)
arr(ax, 6.5, 5.42, 7.1, 3.28, C_GREEN, '', rad=-0.05)

# ── Row 4: Grasp sub-pipeline ─────────────────────────────────────────────
cluster(ax, 0.2, 0.55, 9.5, 2.0, 'Grasp Sub-Pipeline', '#EFF6FF', C_BLUE)

box(ax, 2.0,  1.27, 2.4, 0.75, 'Depth Refinement',  'Tighten bounding box',
    'using depth channel',        fc=C_LBLUE, ec=C_BLUE)
box(ax, 5.2,  1.27, 2.6, 0.75, '3D Pose Estimation', 'Cylinder fitting',
    'Object centre in 3D',        fc=C_LBLUE, ec=C_BLUE)
box(ax, 8.4,  1.27, 2.2, 0.75, 'Arm Motion',         'Motion planning',
    'Grasp execution',            fc=C_LBLUE, ec=C_BLUE)

arr(ax, 4.3, 2.42, 3.2, 2.0,  C_BLUE, '')   # grab_bottle → refinement
arr(ax, 3.2, 1.27, 3.9, 1.27, C_BLUE)
arr(ax, 6.5, 1.27, 7.3, 1.27, C_BLUE)

# ── Safety Monitor ────────────────────────────────────────────────────────
box(ax,11.4, 1.27, 2.5, 0.75,
    'Safety Monitor',
    'Failure detection',
    'Safe recovery mode',         fc=C_LRED, ec=C_RED)

# VLM → safety (dashed)
ax.annotate('', xy=(11.4, 1.65), xytext=(6.5, 5.42),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2,
                            linestyle='dashed',
                            connectionstyle='arc3,rad=-0.25'), zorder=5)
ax.text(10.6, 3.5, 'fail count', fontsize=7.2, color=C_RED, ha='center',
        rotation=-60)

# ── ROS topics footer ─────────────────────────────────────────────────────
ax.text(4.8, 0.30, 'ROS middleware — real-time communication between all components',
        ha='center', va='center', fontsize=8, color=C_SLATE,
        fontstyle='italic')

ax.set_title('EmbodiedAgent — System Architecture',
             fontsize=12, fontweight='bold', color=C_NAVY, pad=6)

fig1.tight_layout(pad=0.3)
fig1.savefig('fig1_arch.pdf')
plt.close(fig1)
print('fig1_arch.pdf  ✓')


# =============================================================================
# FIG 2 — StateManager Ontology  (taller, cleaner)
# =============================================================================
fig2, ax = plt.subplots(figsize=(6.5, 4.8))
ax.set_xlim(0, 6.5); ax.set_ylim(0, 4.8)
ax.axis('off')
fig2.patch.set_facecolor('white')

def onode(ax, x, y, w, h, title, props='', fc='#DBEAFE', ec='#2563EB'):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.12',
                        fc=fc, ec=ec, lw=1.6, zorder=3)
    ax.add_patch(b)
    if props:
        ax.text(x, y+h*0.14, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=ec, zorder=4)
        ax.text(x, y-h*0.22, props, ha='center', va='center',
                fontsize=7.5, color=C_SLATE, zorder=4, style='italic')
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=ec, zorder=4)

def oarr(ax, x0, y0, x1, y1, lbl='', col=C_GRAY, rad=0.0, lw=1.3):
    cs = f'arc3,rad={rad}'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                connectionstyle=cs), zorder=5)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my, lbl, ha='center', va='center', fontsize=7.5, color=col,
                bbox=dict(fc='white', ec='none', pad=1.5, alpha=0.9), zorder=6)

# Top row
onode(ax, 1.3,  4.1, 1.9, 0.72, 'Robot State',
      'gripper · posture · pose',   C_LBLUE,  C_BLUE)
onode(ax, 3.25, 4.1, 2.0, 0.72, 'Detected Object',
      'position · confidence\ntime last seen', C_LGREEN, C_GREEN)
onode(ax, 5.35, 4.1, 1.9, 0.72, 'Scene Graph',
      'object relationships',        C_LPURP,  C_PURPLE)

# Middle row
onode(ax, 1.3,  2.5, 1.9, 0.72, 'Task History',
      'executed skills\n& outcomes',  C_LRED,   C_RED)
onode(ax, 3.25, 2.5, 2.0, 0.72, 'Spatial Relation',
      'left of  ·  right of\nabove  ·  near  ·  on surface', C_LAMBER, C_AMBER)
onode(ax, 5.35, 2.5, 1.9, 0.72, 'Location',
      'table · shelf\nhome · unknown', C_LTEAL, C_TEAL)

# Bottom
onode(ax, 3.25, 0.85, 4.8, 0.62,
      'VLM Prompt Context  (serialized text injected each inference step)',
      fc='#F8FAFC', ec=C_NAVY)

# Arrows
oarr(ax, 2.25, 4.1,  2.25, 4.1+0.001, lbl='')   # dummy for spacing
oarr(ax, 2.25, 4.1,  2.35, 4.1,   'detects',       C_GREEN)
oarr(ax, 4.25, 4.1,  4.45, 4.1,   'hasRelation',   C_PURPLE)
oarr(ax, 5.35, 3.74, 3.25+1.0, 2.86, 'instanceOf', C_AMBER, rad=0.2)
oarr(ax, 3.25, 3.74, 3.25, 2.86,  'involves',      C_AMBER)
oarr(ax, 1.3,  3.74, 1.3,  2.86,  'executed',      C_RED)
oarr(ax, 2.8,  3.74, 5.35-0.4, 2.86, 'locatedAt',  C_TEAL, rad=-0.2)
oarr(ax, 2.25, 2.5,  2.35, 2.5,   'updatedBy',     C_RED)

# Everything serializes to VLM prompt
for sx, sy in [(1.3, 2.14), (3.25, 2.14), (5.35, 2.14)]:
    oarr(ax, sx, sy, 3.25, 1.16, col=C_NAVY, rad=0.0, lw=1.0)

ax.text(3.25, 4.75, 'StateManager — Dynamic Scene Ontology',
        ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=C_NAVY)

fig2.tight_layout(pad=0.2)
fig2.savefig('fig2_ontology.pdf')
plt.close(fig2)
print('fig2_ontology.pdf  ✓')


# =============================================================================
# FIG 3 — Head search scan pattern
# =============================================================================
fig3, ax = plt.subplots(figsize=(4.8, 4.5), subplot_kw=dict(projection='polar'))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)   # clockwise = robot's right

# Positions: (number, label, pan_rad, tilt, colour, marker)
POS = [
    (1,  'center',        0.0,   'level', C_BLUE,   'o'),
    (2,  'center↓',       0.0,  -0.45,   C_BLUE,   's'),
    (3,  'left',          0.8,   'level', C_GREEN,  'o'),
    (4,  'left↓',         0.8,  -0.45,   C_GREEN,  's'),
    (5,  'far left',      1.4,   'level', C_GREEN,  '^'),
    (6,  'far left↓',     1.4,  -0.45,   C_GREEN,  'v'),
    (7,  'right',        -0.8,   'level', C_RED,    'o'),
    (8,  'right↓',       -0.8,  -0.45,   C_RED,    's'),
    (9,  'far right',    -1.4,   'level', C_RED,    '^'),
    (10, 'far right↓',   -1.4,  -0.45,   C_RED,    'v'),
    (11, 'up',            0.0,  +0.30,   C_AMBER,  'D'),
    (12, 'return',        0.0,   'level', C_BLUE,   'P'),
]

def to_pr(pan, tilt):
    if tilt == 'level': r = 1.0
    elif tilt == +0.30: r = 0.55
    else:               r = 1.55
    return pan, r

# Draw scan path
pts = [to_pr(p[2], p[3]) for p in POS]
ax.plot([p[0] for p in pts], [p[1] for p in pts],
        '-', color=C_GRAY, alpha=0.18, lw=1.0, zorder=1)

# Reference circles
for r_c in [1.0, 1.55]:
    ax.plot(np.linspace(0, 2*np.pi, 300), [r_c]*300,
            '--', color=C_GRAY, alpha=0.18, lw=0.7)

# Robot at centre — draw as text annotation in axes coordinates to avoid clipping
ax.plot(0, 0, 'o', color=C_NAVY, ms=18, zorder=10, clip_on=False)
# Use figure-level text anchored to the axes origin
ax.text(0, 0, 'TIAGo', transform=ax.transData,
        ha='center', va='center', fontsize=6.5, color='white',
        fontweight='bold', zorder=11, clip_on=False)

# Draw each position
for num, lbl, pan, tilt, col, mk in POS:
    theta, r = to_pr(pan, tilt)
    # spoke line
    ax.plot([theta, theta], [0.22, r - 0.12], '-', color=col, alpha=0.3, lw=1.1, zorder=2)
    ax.plot(theta, r, mk, color=col, ms=9, zorder=5,
            markeredgecolor='white', markeredgewidth=0.7)
    ax.text(theta, r + 0.28, str(num), ha='center', va='center',
            fontsize=7.5, fontweight='bold', color=col, zorder=6)

# Pan angle labels at outer edge
for lbl, pan_rad in [('Forward\n0 rad', 0), ('Left +0.8', 0.8),
                      ('Far left +1.4', 1.4), ('Right −0.8', -0.8),
                      ('Far right −1.4', -1.4)]:
    ax.text(pan_rad, 2.02, lbl, ha='center', va='bottom', fontsize=6.5,
            color=C_SLATE, multialignment='center')

# Legend
handles = [
    mpatches.Patch(color=C_BLUE,  label='Center / Return  (pos. 1,2,12)'),
    mpatches.Patch(color=C_GREEN, label='Left  (+0.8 / +1.4 rad)'),
    mpatches.Patch(color=C_RED,   label='Right (−0.8 / −1.4 rad)'),
    mpatches.Patch(color=C_AMBER, label='Up    (+0.3 rad tilt, pos. 11)'),
    mpatches.Patch(color=C_GRAY,  label='○ level   ◻ tilt −0.45 rad   ▲ far'),
]
ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.28),
          ncol=2, fontsize=7.2, frameon=True, framealpha=0.95, edgecolor=C_GRAY)

ax.set_rticks([]); ax.set_xticks([]); ax.spines['polar'].set_visible(False)
ax.set_ylim(0, 2.3)
ax.set_title('Head Search: 12 Pan/Tilt Positions\n(top-down view, N = forward)',
             fontsize=10, fontweight='bold', color=C_NAVY, pad=14)

fig3.tight_layout()
fig3.savefig('fig3_headscan.pdf')
plt.close(fig3)
print('fig3_headscan.pdf  ✓')


# =============================================================================
# FIG 4 — Depth bbox refinement (4-panel)
# =============================================================================
import matplotlib.patches as patches2

fig4, axes = plt.subplots(1, 4, figsize=(8.5, 3.0))
fig4.patch.set_facecolor('white')
TITLES = ['(a) VLM output\n(loose bbox)',
          '(b) Depth ROI\n(near cluster)',
          '(c) Refined bbox\n(tight)',
          '(d) 3D point cloud\n(RANSAC centroid)']

for ax in axes:
    ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal'); ax.axis('off')

# (a)
ax = axes[0]
ax.imshow(np.full((100,100,3), [0.88,0.90,0.94]),
          extent=[0,100,0,100], origin='lower', zorder=0)
ax.add_patch(patches2.FancyBboxPatch((37,17),26,60, boxstyle='round,pad=2',
             fc='#C9934A', ec='#7a5c1e', lw=1.3, zorder=2))
ax.add_patch(patches2.FancyBboxPatch((18,7), 64,80, boxstyle='square,pad=0',
             fc='none', ec=C_BLUE, lw=2.3, ls='--', zorder=3))
ax.text(50, 91,'VLM bounding box\n(loose)', ha='center', va='bottom',
        fontsize=8.5, color=C_BLUE, fontweight='bold')
ax.text(50,  3,'covers background region', ha='center', va='bottom',
        fontsize=7,   color=C_GRAY)

# (b)
ax = axes[1]
xg, yg = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
depth = np.clip(0.5 + 0.25*np.sin(xg*4) + 0.2*np.cos(yg*3), 0, 1)
depth[19:80, 32:68] -= 0.4
ax.imshow(np.clip(depth,0,1), extent=[0,100,0,100], origin='lower',
          cmap='RdYlBu_r', alpha=0.92, zorder=0)
ax.add_patch(patches2.Ellipse((50,48),38,62, fc='none',
             ec=C_RED, lw=2.3, zorder=3))
ax.text(50, 84,'Nearest depth cluster\n(foreground only)',   ha='center', va='bottom',
        fontsize=8.5, color=C_RED, fontweight='bold')
ax.text(50,  3,'object closer than background', ha='center', va='bottom',
        fontsize=7,   color=C_GRAY)

# (c)
ax = axes[2]
ax.imshow(np.full((100,100,3), [0.88,0.90,0.94]),
          extent=[0,100,0,100], origin='lower', zorder=0)
ax.add_patch(patches2.FancyBboxPatch((37,17),26,60, boxstyle='round,pad=2',
             fc='#C9934A', ec='#7a5c1e', lw=1.3, zorder=2))
ax.add_patch(patches2.FancyBboxPatch((18,7), 64,80, boxstyle='square,pad=0',
             fc='none', ec=C_BLUE, lw=1.0, ls='--', alpha=0.2, zorder=3))
ax.add_patch(patches2.FancyBboxPatch((32,14),36,68, boxstyle='square,pad=0',
             fc='none', ec=C_RED, lw=2.8, zorder=4))
ax.text(50, 86,'Tight bounding box\n(object only)',  ha='center', va='bottom',
        fontsize=8.5, color=C_RED, fontweight='bold')
ax.text(50,  3,'↓ 88 % smaller area',  ha='center', va='bottom',
        fontsize=7,   color=C_GREEN, fontweight='bold')

# (d)
ax = axes[3]
ax.set_facecolor('#070B14')
rng = np.random.default_rng(7)
ax.scatter(rng.uniform(5,95,230), rng.uniform(5,95,230),
           s=2, c='#1E293B', alpha=0.7, zorder=1)
ang = np.linspace(0, 2*np.pi, 90)
ax.scatter(50+14*np.cos(ang)+rng.normal(0,1.5,90),
           18+64*(ang/(2*np.pi))+rng.normal(0,1.2,90),
           s=8, c='#60A5FA', alpha=0.95, zorder=3)
ax.add_patch(patches2.Ellipse((50,50),30,70, fc='none',
             ec='#93C5FD', lw=1.6, ls=':', zorder=4))
ax.plot(50,50,'*', color='#FBBF24', ms=17, zorder=6,
        markeredgecolor='white', markeredgewidth=0.5)
ax.text(50, 86,'3D object centre\nestimated (★)', ha='center', va='bottom',
        fontsize=8.5, color='#93C5FD', fontweight='bold')
ax.text(50,  3,'cylinder fitting on clean points', ha='center', va='bottom',
        fontsize=7,   color='#64748B')

for ax, t in zip(axes, TITLES):
    ax.set_title(t, fontsize=8.5, fontweight='bold', color=C_NAVY, pad=5)

fig4.suptitle('Depth-based Bounding Box Refinement & 3D Pose Estimation',
              fontsize=11, fontweight='bold', color=C_NAVY, y=1.03)
fig4.tight_layout(pad=0.5)
fig4.savefig('fig4_depth.pdf')
plt.close(fig4)
print('fig4_depth.pdf  ✓')


# =============================================================================
# FIG 5 — Task success rate  (no overlapping labels)
# =============================================================================
scenarios = ['S1\nDirect\nGrasp', 'S2\nSearch+\nGrasp', 'S3\nMulti-obj\nDisamb.',
             'S4\nOpen-end\nSelect', 'S5\nFace ID\n+Greet', 'S6\nMulti-step\nWave+Grasp',
             'Overall']
success   = [93.3, 80.0, 73.3, 86.7, 86.7, 73.3, 82.0]
colours   = [C_GREEN if v >= 80 else C_AMBER for v in success]
colours[-1] = C_NAVY

fig5, ax = plt.subplots(figsize=(8.0, 4.2))
x = np.arange(len(scenarios))
bars = ax.bar(x, success, color=colours, width=0.6,
              edgecolor='white', linewidth=1.3, zorder=3, alpha=0.9)

for bar, v in zip(bars, success):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.9,
            f'{v:.1f}%', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color=bar.get_facecolor())

ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=8.5, multialignment='center')
ax.tick_params(axis='x', pad=4)
ax.set_ylabel('Task Success Rate (%)', fontsize=10)
ax.set_ylim(0, 110)
ax.yaxis.grid(True, ls='--', color='#E5E7EB', zorder=0, alpha=0.8)
ax.set_axisbelow(True)
ax.axhline(80, color=C_GRAY, lw=1.0, ls=':', alpha=0.7)
ax.text(6.55, 81, '80 % target', fontsize=8, color=C_GRAY, ha='right', va='bottom')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color=C_GREEN, label='≥ 80 % (target met)'),
    mpatches.Patch(color=C_AMBER, label='< 80 % (improvement needed)'),
    mpatches.Patch(color=C_NAVY,  label='Overall average'),
]
ax.legend(handles=legend_patches, fontsize=8.5, loc='lower right',
          framealpha=0.95, edgecolor=C_GRAY)
ax.set_title('Task Success Rate per Evaluation Scenario  (n = 15 trials each)',
             fontsize=11, fontweight='bold', color=C_NAVY, pad=10)

fig5.subplots_adjust(bottom=0.22)
fig5.savefig('fig5_tasks.pdf')
plt.close(fig5)
print('fig5_tasks.pdf  ✓')


# =============================================================================
# FIG 6 — Grouped bars: grasp + 3D error
# =============================================================================
objects   = ['Bottle', 'Cup', 'Box', 'Remote', 'Phone']
without_g = [51.7, 48.3, 61.7, 43.3, 55.0]
with_g    = [81.7, 75.0, 86.7, 68.3, 78.3]
without_e = [ 8.3,  9.1,  6.5, 11.2,  7.8]
with_e    = [ 2.1,  2.8,  1.9,  3.5,  2.4]
x = np.arange(len(objects)); w = 0.36

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

for ax_, base, enhanced, ylabel, title, col in [
    (ax1, without_g, with_g,    'Grasp Success Rate (%)',      'Grasp Success Rate',   C_BLUE),
    (ax2, without_e, with_e,    'Mean 3D Position Error (cm)', '3D Position Error (cm)', C_GREEN),
]:
    b1 = ax_.bar(x-w/2, base,     w, label='Without refinement',
                 color='#94A3B8', ec='white', lw=1.2, zorder=3, alpha=0.85)
    b2 = ax_.bar(x+w/2, enhanced, w, label='With depth refinement',
                 color=col,       ec='white', lw=1.2, zorder=3, alpha=0.9)
    for bar in b1:
        ax_.text(bar.get_x()+bar.get_width()/2, bar.get_height() + (0.8 if ax_ is ax1 else 0.15),
                 f'{bar.get_height():.0f}' if ax_ is ax1 else f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, color='#64748B')
    for bar in b2:
        ax_.text(bar.get_x()+bar.get_width()/2, bar.get_height() + (0.8 if ax_ is ax1 else 0.15),
                 f'{bar.get_height():.0f}' if ax_ is ax1 else f'{bar.get_height():.1f}',
                 ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')
    ax_.set_xticks(x); ax_.set_xticklabels(objects, fontsize=9.5)
    ax_.set_ylabel(ylabel, fontsize=9.5)
    ax_.set_ylim(0, (105 if ax_ is ax1 else 14.5))
    ax_.yaxis.grid(True, ls='--', color='#E5E7EB', zorder=0)
    ax_.set_axisbelow(True)
    ax_.legend(fontsize=8.5, framealpha=0.95, edgecolor=C_GRAY)
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
    ax_.set_title(title, fontsize=10, fontweight='bold', color=C_NAVY)

fig6.suptitle('Impact of Depth-based Bounding Box Refinement on Grasping Performance',
              fontsize=11, fontweight='bold', color=C_NAVY)
fig6.tight_layout(pad=1.2)
fig6.savefig('fig6_grasp.pdf')
plt.close(fig6)
print('fig6_grasp.pdf  ✓')

print('\nAll figures generated.')
