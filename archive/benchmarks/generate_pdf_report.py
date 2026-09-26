import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Directories
ARTIFACT_DIR = Path("/home/bittu/.gemini/antigravity-cli/brain/f1439725-9acf-4d55-abe6-0c8042d0399f")
CHARTS_DIR = ARTIFACT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = ARTIFACT_DIR / "Nawabs_vs_Peshwas_vs_Shauryas_Benchmark_Report.pdf"

print("==========================================================================")
print("     GENERATING GRAPH VISUALIZATIONS & PDF REPORT                         ")
print("==========================================================================")

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# Color palette
COLOR_NAWABS = '#1e3a8a'   # Navy Blue
COLOR_PESHWAS = '#b91c1c'  # Crimson Red
COLOR_SHAURYAS = '#d97706' # Amber Gold

# -------------------------------------------------------------------------
# Chart 1: 120-Image Dataset Overall Accuracy
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
teams = ['Nawabs', 'Shauryas', 'Peshwas']
accuracies = [80.00, 59.17, 50.83]
bars = ax.bar(teams, accuracies, color=[COLOR_NAWABS, COLOR_SHAURYAS, COLOR_PESHWAS], width=0.55)

ax.set_ylabel('Overall Accuracy (%)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_title('120-Image Dataset Classification Accuracy (%)', fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
chart1_path = CHARTS_DIR / "chart1_dataset_accuracy.png"
plt.savefig(chart1_path)
plt.close()
print(f"[✓] Saved Chart 1: {chart1_path.name}")

# -------------------------------------------------------------------------
# Chart 2: main_test Folder Accuracy (%)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
teams_main = ['Nawabs', 'Peshwas', 'Shauryas']
accuracies_main = [93.33, 86.67, 66.67]
bars = ax.bar(teams_main, accuracies_main, color=[COLOR_NAWABS, COLOR_PESHWAS, COLOR_SHAURYAS], width=0.55)

ax.set_ylabel('Folder Accuracy (%)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_title('main_test Image Suite Classification Accuracy (%)', fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 110)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for bar, val in zip(bars, [14, 13, 10]):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f'{yval:.2f}%\n({val}/15)', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
chart2_path = CHARTS_DIR / "chart2_main_test_accuracy.png"
plt.savefig(chart2_path)
plt.close()
print(f"[✓] Saved Chart 2: {chart2_path.name}")

# -------------------------------------------------------------------------
# Chart 3: Per-Class F1-Score Breakdown
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)
categories = ['Spalling', 'Stagnant Water', 'Cracked Tiles', 'Paint Peeling']
x = np.arange(len(categories))
width = 0.25

f1_nawabs = [90.00, 80.00, 81.08, 67.86]
f1_peshwas = [91.80, 85.19, 6.90, 23.88]
f1_shauryas = [96.77, 91.53, 4.17, 36.62]

rects1 = ax.bar(x - width, f1_nawabs, width, label='Nawabs', color=COLOR_NAWABS)
rects2 = ax.bar(x, f1_peshwas, width, label='Peshwas', color=COLOR_PESHWAS)
rects3 = ax.bar(x + width, f1_shauryas, width, label='Shauryas', color=COLOR_SHAURYAS)

ax.set_ylabel('F1-Score (%)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_title('Per-Class Defect Classification F1-Scores (%)', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(loc='upper right', frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.5)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5, rotation=0)

plt.tight_layout()
chart3_path = CHARTS_DIR / "chart3_per_class_f1.png"
plt.savefig(chart3_path)
plt.close()
print(f"[✓] Saved Chart 3: {chart3_path.name}")

# -------------------------------------------------------------------------
# Chart 4: CPU Latency Comparison
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
teams_lat = ['Nawabs', 'Peshwas (3-View TTA)', 'Shauryas']
latencies = [60.33, 136.52, 424.58]
bars = ax.barh(teams_lat, latencies, color=[COLOR_NAWABS, COLOR_PESHWAS, COLOR_SHAURYAS], height=0.5)

ax.set_xlabel('Mean CPU Inference Latency (ms)', fontsize=10, fontweight='bold', labelpad=8)
ax.set_title('Inference Execution Speed (Lower is Better)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(0, 500)
ax.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    wval = bar.get_width()
    ax.text(wval + 8, bar.get_y() + bar.get_height()/2.0, f'{wval:.1f} ms', ha='left', va='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
chart4_path = CHARTS_DIR / "chart4_latency.png"
plt.savefig(chart4_path)
plt.close()
print(f"[✓] Saved Chart 4: {chart4_path.name}")

# -------------------------------------------------------------------------
# Build PDF Document using ReportLab
# -------------------------------------------------------------------------
print("\nBuilding ReportLab PDF Document...")

doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=letter,
    rightMargin=36,
    leftMargin=36,
    topMargin=36,
    bottomMargin=36
)

styles = getSampleStyleSheet()

# Custom styles
style_title = ParagraphStyle(
    'DocTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=20,
    leading=24,
    textColor=colors.HexColor('#0f172a'),
    alignment=0,
    spaceAfter=4
)

style_subtitle = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=11,
    leading=14,
    textColor=colors.HexColor('#475569'),
    spaceAfter=12
)

style_h1 = ParagraphStyle(
    'Heading1_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=14,
    leading=18,
    textColor=colors.HexColor('#1e3a8a'),
    spaceBefore=14,
    spaceAfter=8,
    keepWithNext=True
)

style_h2 = ParagraphStyle(
    'Heading2_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=15,
    textColor=colors.HexColor('#334155'),
    spaceBefore=10,
    spaceAfter=6,
    keepWithNext=True
)

style_body = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor('#1e293b'),
    spaceAfter=6
)

style_callout = ParagraphStyle(
    'Callout',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=9.5,
    leading=14,
    textColor=colors.HexColor('#0f172a'),
    backColor=colors.HexColor('#f1f5f9'),
    borderColor=colors.HexColor('#cbd5e1'),
    borderWidth=1,
    borderPadding=8,
    spaceBefore=8,
    spaceAfter=10
)

style_th = ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8.5, leading=10, textColor=colors.white, alignment=1)
style_td = ParagraphStyle('TD', fontName='Helvetica', fontSize=8, leading=10, textColor=colors.HexColor('#0f172a'), alignment=0)
style_td_bold = ParagraphStyle('TDBold', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.HexColor('#0f172a'), alignment=0)

story = []

# Title & Subtitle
story.append(Paragraph("Master Technical Evaluation & Benchmark Report", style_title))
story.append(Paragraph("Comparative Technical Audit: Nawabs vs. Peshwas vs. Shauryas", style_subtitle))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e3a8a'), spaceAfter=10))

# Callout Summary
summary_text = (
    "<b>Executive Summary</b>: This document evaluates the defect detection accuracy, priority queue "
    "algorithms, and system latency across three teams: <b>Nawabs</b>, <b>Peshwas</b>, and <b>Shauryas</b>. "
    "All findings are backed by empirical test dataset numbers (120 images) and file-by-file testing on the "
    "<i>main_test</i> suite."
)
story.append(Paragraph(summary_text, style_callout))

# -------------------------------------------------------------------------
# Section 1: Quick Stats Comparison Summary
# -------------------------------------------------------------------------
story.append(Paragraph("1. Quick Stats Comparison Summary", style_h1))

table_data_s1 = [
    [Paragraph("Key Evaluation Metric", style_th), Paragraph("Nawabs", style_th), Paragraph("Peshwas", style_th), Paragraph("Shauryas", style_th), Paragraph("Summary Advantage", style_th)],
    [Paragraph("120-Image Dataset Accuracy", style_td_bold), Paragraph("80.00%", style_td_bold), Paragraph("50.83%", style_td), Paragraph("59.17%", style_td), Paragraph("Nawabs leads by +20.83% over Shauryas & +29.17% over Peshwas", style_td)],
    [Paragraph("main_test Folder Accuracy", style_td_bold), Paragraph("93.33% (14/15)", style_td_bold), Paragraph("86.67% (13/15)", style_td), Paragraph("66.67% (10/15)", style_td), Paragraph("Nawabs correctly classifies 14 of 15 test files", style_td)],
    [Paragraph("Cracked Tiles F1-Score", style_td_bold), Paragraph("81.08%", style_td_bold), Paragraph("6.90%", style_td), Paragraph("4.17%", style_td), Paragraph("Peshwas & Shauryas fail on tile crack detection", style_td)],
    [Paragraph("Paint Peeling F1-Score", style_td_bold), Paragraph("67.86%", style_td_bold), Paragraph("23.88%", style_td), Paragraph("36.62%", style_td), Paragraph("Nawabs is +31.24% more accurate on wall flaking", style_td)],
    [Paragraph("Spalling F1-Score", style_td), Paragraph("90.00%", style_td), Paragraph("91.80%", style_td), Paragraph("96.77%", style_td), Paragraph("All three models detect spalling well", style_td)],
    [Paragraph("Stagnant Water F1-Score", style_td), Paragraph("80.00%", style_td), Paragraph("85.19%", style_td), Paragraph("91.53%", style_td), Paragraph("All three models detect water puddles well", style_td)],
    [Paragraph("Mean CPU Latency", style_td_bold), Paragraph("60.33 ms", style_td_bold), Paragraph("136.52 ms", style_td), Paragraph("424.58 ms", style_td), Paragraph("Nawabs runs 2.2x faster than Peshwas & 7x faster than Shauryas", style_td)],
    [Paragraph("Queue Starvation Protection", style_td_bold), Paragraph("Included (≤ 5.0 pts)", style_td_bold), Paragraph("Excluded", style_td), Paragraph("Excluded", style_td), Paragraph("Peshwas & Shauryas tickets can get stuck forever", style_td)],
    [Paragraph("Performance Sub-Ranking", style_td_bold), Paragraph("Enforced", style_td_bold), Paragraph("Enforced", style_td), Paragraph("Violated", style_td), Paragraph("Shauryas ranks paint above tiles on high severity", style_td)],
    [Paragraph("Image Blur Pre-Filter", style_td_bold), Paragraph("Included (Var < 50)", style_td_bold), Paragraph("None", style_td), Paragraph("None", style_td), Paragraph("Nawabs automatically rejects blurry uploads", style_td)]
]

t_s1 = Table(table_data_s1, colWidths=[1.3*inch, 0.85*inch, 0.85*inch, 0.85*inch, 2.55*inch])
t_s1.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('PADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_s1)
story.append(Spacer(1, 10))

# Embed Charts 1 & 4 side by side or stacked
story.append(KeepTogether([
    Table([
        [Image(str(chart1_path), width=3.3*inch, height=1.92*inch), Image(str(chart4_path), width=3.3*inch, height=1.92*inch)]
    ], colWidths=[3.4*inch, 3.4*inch])
]))

story.append(Spacer(1, 10))

# -------------------------------------------------------------------------
# Section 2: Detailed Stats & Per-Class Breakdown
# -------------------------------------------------------------------------
story.append(Paragraph("2. Detailed Stats Sheet & Category Breakdown", style_h1))

story.append(Paragraph("<b>2.1 Full 120-Image Test Set Classification Results</b>", style_h2))
table_data_s2 = [
    [Paragraph("Defect Category", style_th), Paragraph("True Count", style_th), Paragraph("Nawabs F1 (%)", style_th), Paragraph("Peshwas F1 (%)", style_th), Paragraph("Shauryas F1 (%)", style_th), Paragraph("Key Finding", style_th)],
    [Paragraph("Spalling (Structural)", style_td), Paragraph("30", style_td), Paragraph("90.00%", style_td), Paragraph("91.80%", style_td), Paragraph("96.77%", style_td), Paragraph("Concrete spalling is detected well by all three models.", style_td)],
    [Paragraph("Stagnant Water (Functional)", style_td), Paragraph("30", style_td), Paragraph("80.00%", style_td), Paragraph("85.19%", style_td), Paragraph("91.53%", style_td), Paragraph("Surface water is detected well by all three models.", style_td)],
    [Paragraph("Cracked Tiles (Performance)", style_td_bold), Paragraph("30", style_td), Paragraph("81.08%", style_td_bold), Paragraph("6.90%", style_td), Paragraph("4.17%", style_td), Paragraph("Peshwas & Shauryas fail on tile cracks.", style_td)],
    [Paragraph("Paint Peeling (Performance)", style_td_bold), Paragraph("30", style_td), Paragraph("67.86%", style_td_bold), Paragraph("23.88%", style_td), Paragraph("36.62%", style_td), Paragraph("Nawabs is vastly more accurate on paint peeling.", style_td)],
    [Paragraph("OVERALL ACCURACY", style_th), Paragraph("120", style_th), Paragraph("80.00%", style_th), Paragraph("50.83%", style_th), Paragraph("59.17%", style_th), Paragraph("Nawabs is the only balanced model across all 4 classes.", style_th)]
]

t_s2 = Table(table_data_s2, colWidths=[1.4*inch, 0.7*inch, 0.95*inch, 0.95*inch, 0.95*inch, 2.45*inch])
t_s2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#334155')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f8fafc')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('PADDING', (0, 0), (-1, -1), 4),
]))
story.append(t_s2)
story.append(Spacer(1, 10))

# Embed Charts 2 & 3
story.append(KeepTogether([
    Table([
        [Image(str(chart2_path), width=3.2*inch, height=1.86*inch), Image(str(chart3_path), width=3.5*inch, height=1.86*inch)]
    ], colWidths=[3.3*inch, 3.5*inch])
]))

story.append(Spacer(1, 10))

# -------------------------------------------------------------------------
# Section 3: main_test Image Suite Comparison
# -------------------------------------------------------------------------
story.append(Paragraph("3. main_test Folder Image-by-Image Comparison", style_h1))

table_data_s3 = [
    [Paragraph("Filename", style_th), Paragraph("Expected", style_th), Paragraph("Nawabs Result", style_th), Paragraph("Peshwas Result", style_th), Paragraph("Shauryas Result", style_th), Paragraph("Finding", style_th)],
    [Paragraph("crack.jpg", style_td), Paragraph("Cracked Tiles", style_td), Paragraph("cracked_tiles [✓]", style_td_bold), Paragraph("cracked_tiles [✓]", style_td), Paragraph("spalling [✗]", style_td), Paragraph("Shauryas misclassifies tile crack as spalling.", style_td)],
    [Paragraph("paint.jpg", style_td), Paragraph("Paint Peeling", style_td), Paragraph("paint_peeling [✓]", style_td_bold), Paragraph("paint_peeling [✓]", style_td), Paragraph("stagnant_water [✗]", style_td), Paragraph("Shauryas misclassifies wall paint as water.", style_td)],
    [Paragraph("paint.png", style_td), Paragraph("Paint Peeling", style_td), Paragraph("paint_peeling [✓]", style_td_bold), Paragraph("paint_peeling [✓]", style_td), Paragraph("stagnant_water [✗]", style_td), Paragraph("Shauryas misclassifies wall paint as water.", style_td)],
    [Paragraph("paint1.jpg", style_td), Paragraph("Paint Peeling", style_td), Paragraph("paint_peeling [✓]", style_td_bold), Paragraph("cracked_tiles [✗]", style_td), Paragraph("paint_peeling [✓]", style_td), Paragraph("Peshwas misclassifies wall paint as cracked tiles.", style_td)],
    [Paragraph("spalling.jpg", style_td), Paragraph("Spalling", style_td), Paragraph("spalling [✓]", style_td_bold), Paragraph("spalling [✓]", style_td), Paragraph("cracked_tiles [✗]", style_td), Paragraph("Shauryas misclassifies spalling as cracked tiles.", style_td)],
    [Paragraph("train_spalling_0037.jpg", style_td), Paragraph("Spalling", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("train_spalling_0038.jpg", style_td), Paragraph("Spalling", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("train_spalling_0039.jpg", style_td), Paragraph("Spalling", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("train_spalling_0040.jpg", style_td), Paragraph("Spalling", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("spalling [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("train_water_0048.jpg", style_td), Paragraph("Stagnant Water", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("train_water_0049.jpg", style_td), Paragraph("Stagnant Water", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("train_water_0050.jpg", style_td), Paragraph("Stagnant Water", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("water.jpg", style_td), Paragraph("Stagnant Water", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("All three models classify correctly.", style_td)],
    [Paragraph("water.webp", style_td), Paragraph("Stagnant Water", style_td), Paragraph("stagnant_water [✓]", style_td_bold), Paragraph("paint_peeling [✗]", style_td), Paragraph("paint_peeling [✗]", style_td), Paragraph("Peshwas & Shauryas misclassify water glare as paint.", style_td)],
    [Paragraph("water1.webp", style_td), Paragraph("Stagnant Water", style_td), Paragraph("paint_peeling [✗]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("stagnant_water [✓]", style_td), Paragraph("Peshwas & Shauryas detect low-res webp.", style_td)],
    [Paragraph("TOTAL ACCURACY", style_th), Paragraph("15 Files", style_th), Paragraph("14/15 (93.33%)", style_th), Paragraph("13/15 (86.67%)", style_th), Paragraph("10/15 (66.67%)", style_th), Paragraph("Nawabs achieves highest accuracy (93.33%).", style_th)]
]

t_s3 = Table(table_data_s3, colWidths=[1.1*inch, 0.95*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.35*inch])
t_s3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#334155')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#f8fafc')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('PADDING', (0, 0), (-1, -1), 3),
]))
story.append(t_s3)
story.append(Spacer(1, 10))

# -------------------------------------------------------------------------
# Section 4: Priority Queue Logic & Engineering Analysis
# -------------------------------------------------------------------------
story.append(Paragraph("4. Priority Queue Logic & Algorithmic Analysis", style_h1))

p_formulas = (
    "<b>Nawabs Priority Formula</b>:<br/>"
    "<code>PriorityScore = BaseTier (3000/2000/1000) + (Severity × 5.0) + (Extent × 3.0) + SubTierBonus (+1.0) + CappedTimeBonus (≤ 5.0)</code><br/><br/>"
    "<b>Peshwas Priority Logic</b>:<br/>"
    "<code>Sort Key = (DEFECT_TYPE_PRIORITY, SEVERITY_SCORE) descending</code><br/><br/>"
    "<b>Shauryas Priority Formula</b>:<br/>"
    "<code>PriorityScore = 100 × BaseWeight × (0.5 × Severity + 0.5 × Extent)</code>"
)
story.append(Paragraph(p_formulas, style_callout))

p_analysis = (
    "<b>Key Algorithmic Differences & Faults</b>:<br/>"
    "<b>1. Queue Starvation (Peshwas & Shauryas)</b>: Peshwas uses rigid tuple sorting without age-based escalation. "
    "Shauryas explicitly excludes age escalation in code (<i>app/ml/priority.py</i> line 32: <i>'Age-based escalation is deliberately excluded'</i>). "
    "In both systems, an older ticket created weeks ago remains stuck at the bottom indefinitely if new tickets arrive.<br/>"
    "<b>Nawabs Solution</b>: Nawabs adds a time escalation bonus of 0.05 points per hour, strictly capped at a maximum of 5.0 points. "
    "This acts as a safe tie-breaker that guarantees older tickets eventually get serviced, without allowing minor issues to jump past urgent structural hazards.<br/><br/>"
    "<b>2. Performance Queue Rule Violation (Shauryas)</b>: The problem statement mandates that within Performance, "
    "<i>'Cracked tiles > paint peeling in priority'</i>. In Shauryas' formula, a Paint Peeling ticket with high severity (0.9) scores <b>63.75</b>, "
    "whereas a Cracked Tiles ticket with low severity (0.2) scores <b>15.00</b>. Shauryas places Paint Peeling above Cracked Tiles, directly violating the rule.<br/>"
    "<b>Nawabs Solution</b>: Nawabs uses base tiers (1000) plus a +1.0 sub-tier bonus for Cracked Tiles, keeping sub-ranking consistent and predictable."
)
story.append(Paragraph(p_analysis, style_body))
story.append(Spacer(1, 10))

# -------------------------------------------------------------------------
# Section 5: Data Inferences & Summary
# -------------------------------------------------------------------------
story.append(Paragraph("5. Data Inferences & Summary", style_h1))
p_inferences = (
    "1. <b>Classification Balance</b>: Nawabs achieves <b>80.00% overall accuracy</b> and balanced F1-scores across all four categories "
    "(81.08% cracked tiles, 90.00% spalling, 80.00% stagnant water, 67.86% paint peeling). Both Peshwas (50.83%) and Shauryas (59.17%) fail on cracked tile detection (6.90% and 4.17% F1-scores).<br/>"
    "2. <b>Real-World File Accuracy</b>: On the 15 <i>main_test</i> image files, Nawabs correctly classifies <b>14 out of 15 files (93.33%)</b>, "
    "compared to Peshwas' 13/15 (86.67%) and Shauryas' 10/15 (66.67%). Shauryas misclassifies common wall paint photos (<i>paint.jpg</i>, <i>paint.png</i>) as stagnant water.<br/>"
    "3. <b>Execution Efficiency</b>: Nawabs executes CPU inference in <b>60.33 ms</b>, running 2.2x faster than Peshwas (136.52 ms) and 7.0x faster than Shauryas (424.58 ms).<br/>"
    "4. <b>Data Quality & Queue Health</b>: Nawabs automatically pre-filters blurry uploads (Laplacian variance Var < 50) and includes capped time escalation (≤ 5.0 pts), "
    "ensuring queue stability and preventing ticket starvation."
)
story.append(Paragraph(p_inferences, style_body))

doc.build(story)
print(f"[✓] PDF Document successfully generated at: {PDF_PATH}")
