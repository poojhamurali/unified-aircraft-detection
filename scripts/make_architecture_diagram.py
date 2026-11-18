"""Generate a publication-grade architecture flow chart for the unified model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure(figsize=(14, 9), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

# Colors
c_input = "#E0F2FE"       # light blue
c_input_border = "#0284C7"
c_branch_cond = "#FEF3C7"  # light amber
c_branch_border = "#D97706"
c_yolo = "#F3E8FF"         # light purple
c_yolo_border = "#7E22CE"
c_film = "#FCE7F3"         # light pink
c_film_border = "#BE185D"
c_out = "#DCFCE7"          # light green
c_out_border = "#15803D"
c_bg_group = "#F8FAFC"
c_bg_border = "#CBD5E1"

def draw_box(ax, x, y, w, h, title, subtitle="", bg="#FFFFFF", border="#000000", lw=1.5, radius=0.25):
    box = patches.FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.0,rounding_size={radius}",
                                linewidth=lw, edgecolor=border, facecolor=bg)
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w/2, y + h/2 + 0.18, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1E293B")
        ax.text(x + w/2, y + h/2 - 0.22, subtitle, ha="center", va="center",
                fontsize=8.5, color="#475569")
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1E293B")

def draw_arrow(ax, x1, y1, x2, y2, label="", color="#475569", lw=2, label_offset=(0, 0.15)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15))
    if label:
        ax.text((x1 + x2)/2 + label_offset[0], (y1 + y2)/2 + label_offset[1], label,
                ha="center", va="center", fontsize=8.5, fontweight="semibold", color=color)

# Title
ax.text(7, 8.6, "Quality- & Modality-Conditioned Unified YOLO Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
ax.text(7, 8.25, "Adaptive single-input aircraft detection across Optical, Infrared, and SAR imagery",
        ha="center", va="center", fontsize=9.5, color="#475569")

# 1. INPUT
draw_box(ax, 0.5, 3.75, 2.0, 1.3, "Input Image", "Optical | IR | SAR\n(640×640×3)",
         bg=c_input, border=c_input_border, lw=2)

# SPLIT ARROWS from input
# Split node
ax.plot([2.5, 2.9], [4.4, 4.4], color="#334155", lw=2)
# Up branch to conditioning
ax.plot([2.9, 2.9], [4.4, 6.8], color="#334155", lw=2)
draw_arrow(ax, 2.9, 6.8, 3.4, 6.8, color="#334155")
# Down branch to YOLO
ax.plot([2.9, 2.9], [4.4, 2.0], color="#334155", lw=2)
draw_arrow(ax, 2.9, 2.0, 3.4, 2.0, color="#334155")

# ----------------- TOP BRANCH: CONDITIONING PIPELINE ----------------- #
# Group Box for Conditioning
group_cond = patches.FancyBboxPatch((3.2, 5.2), 6.5, 2.6,
                                    boxstyle="round,pad=0.1,rounding_size=0.3",
                                    linewidth=1, linestyle="--", edgecolor=c_branch_border, facecolor="#FFFBEB")
ax.add_patch(group_cond)
ax.text(3.4, 7.6, "BRANCH 1: Modality & Quality Conditioning", fontsize=9, fontweight="bold", color=c_branch_border)

# QualityNet Box
draw_box(ax, 3.4, 6.2, 2.7, 1.25, "QualityNet", "8 Image Stats + 3-layer CNN\n(Contrast, Speckle, Detail...)",
         bg=c_branch_cond, border=c_branch_border)

# Modality Token Box
draw_box(ax, 3.4, 5.35, 2.7, 0.7, "Modality Token", "Learned sensor embedding (32d)",
         bg=c_branch_cond, border=c_branch_border)

# Arrow from QualityNet to MLP (placed higher so labels do not collide)
draw_arrow(ax, 6.1, 6.82, 6.9, 6.82)
ax.text(6.5, 7.02, "Quality vector (32d)", ha="center", va="bottom", fontsize=8, fontweight="semibold", color="#B45309")

# Arrow from Modality Token to MLP
draw_arrow(ax, 6.1, 5.7, 6.9, 5.7)
ax.text(6.5, 5.48, "Sensor token (32d)", ha="center", va="top", fontsize=8, fontweight="semibold", color="#B45309")

# Conditioning MLP Box
draw_box(ax, 6.9, 5.4, 2.7, 1.8, "Conditioning MLP", "Concatenate (64d)\n→ Linear + SiLU\n→ Cond Vector (128d)",
         bg="#FDE68A", border=c_branch_border, lw=2)

# ----------------- BOTTOM BRANCH: YOLO FEATURE EXTRACTION ----------------- #
# Group Box for YOLO
group_yolo = patches.FancyBboxPatch((3.2, 0.95), 6.5, 3.7,
                                   boxstyle="round,pad=0.1,rounding_size=0.3",
                                   linewidth=1, linestyle="--", edgecolor=c_yolo_border, facecolor="#FAF5FF")
ax.add_patch(group_yolo)
ax.text(3.4, 4.4, "BRANCH 2: Shared YOLOv8s Backbone & Neck", fontsize=9, fontweight="bold", color=c_yolo_border)

# YOLO Layer 1
draw_box(ax, 3.4, 1.6, 1.8, 1.3, "YOLO Layer i", "Conv / C2f / SPPF\n(Original feature X)",
         bg=c_yolo, border=c_yolo_border)

# Arrow to FiLM
draw_arrow(ax, 5.2, 2.25, 6.0, 2.25, label="Original X", label_offset=(0, 0.2))

# FiLM Wrapper Box
draw_box(ax, 6.0, 1.35, 3.5, 1.8, "FiLM Adapter Module",
         "Generate (γ, β) per channel from cond\nAdjusted = X · (1 + tanh γ) + β\n[Applied at 8 Backbone/PAN layers]",
         bg=c_film, border=c_film_border, lw=2)

# Arrow from MLP to FiLM (The key conditioning arrow)
ax.plot([8.25, 8.25], [5.4, 3.5], color=c_film_border, lw=2.5, linestyle=":")
draw_arrow(ax, 8.25, 3.5, 8.25, 3.15, color=c_film_border, lw=2.5)
ax.text(8.35, 4.25, "Scale (γ) &\nShift (β)\ninstructions", ha="left", va="center",
        fontsize=8.5, fontweight="bold", color=c_film_border)

# Arrow from FiLM to Next YOLO Layers (clean, well-spaced)
draw_arrow(ax, 7.75, 1.35, 7.75, 0.65, color=c_yolo_border, lw=2)
ax.plot([7.75, 4.3], [0.65, 0.65], color=c_yolo_border, lw=2)
ax.plot([4.3, 4.3], [0.65, 1.35], color=c_yolo_border, lw=2)
ax.text(6.0, 0.45, "Adjusted features replace original & feed subsequent layers",
        fontsize=8.2, ha="center", style="italic", color="#6B21A8", fontweight="medium")

# ----------------- OUTPUT STAGE ----------------- #
# Arrow from YOLO/FiLM to Detection Head
draw_arrow(ax, 9.5, 2.25, 10.4, 2.25, color="#334155")
ax.text(9.95, 2.5, "Deep adapted\nfeatures", ha="center", va="bottom",
        fontsize=8, fontweight="semibold", color="#334155")

# Single Shared Detection Head
draw_box(ax, 10.4, 1.55, 3.0, 1.4, "Shared Detection Head", "Ultralytics Detect Head\nClass 0: 'aircraft'\n(One head for all sensors)",
         bg=c_yolo, border=c_yolo_border, lw=2)

# Final Prediction Box
draw_arrow(ax, 11.9, 2.95, 11.9, 3.65, color=c_out_border, lw=2)
draw_box(ax, 10.4, 3.65, 3.0, 1.5, "Final Output", "Bounding Boxes + Confidence\n✓ Optical Planes\n✓ Infrared Air/Drones\n✓ SAR Aircraft",
         bg=c_out, border=c_out_border, lw=2.5)

# Explanatory callout footer
footer = patches.FancyBboxPatch((0.5, 0.05), 12.9, 0.35,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1)
ax.add_patch(footer)
ax.text(7.0, 0.22, "Key Principle: Single shared YOLO weights across all 3 sensors; FiLM modules dynamically scale & shift feature maps based on image condition and sensor type.",
        ha="center", va="center", fontsize=8.2, color="#334155", fontweight="medium")

plt.tight_layout()
out = Path(__file__).resolve().parents[1] / "outputs" / "unified_architecture_flowchart.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight", dpi=300)
print(f"Saved flow chart to {out}")
