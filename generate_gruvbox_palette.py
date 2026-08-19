import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import gruvbox_theme
from gruvbox_theme import GRUVBOX, gruvbox_rainbow_cmap, gruvbox_heat_cmap, gruvbox_cool_cmap

def create_gruvbox_palette_image(output_path="gruvbox.png"):
    # High resolution canvas with generous vertical height
    fig = plt.figure(figsize=(15, 12), dpi=300)
    fig.patch.set_facecolor(GRUVBOX["bg0"])
    
    # Header
    fig.text(0.5, 0.965, "Gruvbox Theme Palette & Colormaps", 
             fontsize=22, fontweight="bold", color=GRUVBOX["fg0"], 
             ha="center", va="top")
    fig.text(0.5, 0.935, "Reference palette for Raman Plotting Pipeline", 
             fontsize=11.5, color=GRUVBOX["fg4"], 
             ha="center", va="top", style="italic")

    # Categories to plot
    bright_accents = [
        ("red", GRUVBOX["red"]),
        ("green", GRUVBOX["green"]),
        ("yellow", GRUVBOX["yellow"]),
        ("blue", GRUVBOX["blue"]),
        ("purple", GRUVBOX["purple"]),
        ("aqua", GRUVBOX["aqua"]),
        ("orange", GRUVBOX["orange"]),
        ("gray", GRUVBOX["gray"]),
    ]

    dark_accents = [
        ("red_dark", GRUVBOX["red_dark"]),
        ("green_dark", GRUVBOX["green_dark"]),
        ("yellow_dark", GRUVBOX["yellow_dark"]),
        ("blue_dark", GRUVBOX["blue_dark"]),
        ("purple_dark", GRUVBOX["purple_dark"]),
        ("aqua_dark", GRUVBOX["aqua_dark"]),
        ("orange_dark", GRUVBOX["orange_dark"]),
        ("gray_dark", GRUVBOX["gray_dark"]),
    ]

    backgrounds = [
        ("bg0_h", GRUVBOX["bg0_h"]),
        ("bg0", GRUVBOX["bg0"]),
        ("bg0_s", GRUVBOX["bg0_s"]),
        ("bg1", GRUVBOX["bg1"]),
        ("bg2", GRUVBOX["bg2"]),
        ("bg3", GRUVBOX["bg3"]),
        ("bg4", GRUVBOX["bg4"]),
    ]

    foregrounds = [
        ("fg0", GRUVBOX["fg0"]),
        ("fg1 (fg)", GRUVBOX["fg1"]),
        ("fg2", GRUVBOX["fg2"]),
        ("fg3", GRUVBOX["fg3"]),
        ("fg4", GRUVBOX["fg4"]),
    ]

    # Clean gridspec layout with distinct, spacious rows
    gs = fig.add_gridspec(
        nrows=4, ncols=6,
        height_ratios=[1.0, 1.0, 1.1, 0.95],
        left=0.05, right=0.95, top=0.90, bottom=0.04,
        hspace=0.75, wspace=0.35
    )

    # 1. Bright / Light Accents
    ax_bright = fig.add_subplot(gs[0, :])
    ax_bright.set_xlim(0, len(bright_accents))
    ax_bright.set_ylim(-0.1, 1.1)
    ax_bright.axis("off")
    ax_bright.text(0, 1.08, "Bright / Light Accents", fontsize=13.5, fontweight="bold", color=GRUVBOX["fg0"], va="bottom")

    for i, (name, hex_code) in enumerate(bright_accents):
        rect = patches.FancyBboxPatch(
            (i + 0.1, 0.42), 0.8, 0.58,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=hex_code, edgecolor=GRUVBOX["bg4"], linewidth=1.2
        )
        ax_bright.add_patch(rect)
        ax_bright.text(i + 0.5, 0.28, name, ha="center", va="top", color=GRUVBOX["fg1"], fontsize=9.5, fontweight="bold")
        ax_bright.text(i + 0.5, 0.06, hex_code, ha="center", va="top", color=GRUVBOX["fg4"], fontsize=8.5, fontfamily="monospace")

    # 2. Normal / Dark Accents
    ax_dark = fig.add_subplot(gs[1, :])
    ax_dark.set_xlim(0, len(dark_accents))
    ax_dark.set_ylim(-0.1, 1.1)
    ax_dark.axis("off")
    ax_dark.text(0, 1.08, "Normal / Dark Accents", fontsize=13.5, fontweight="bold", color=GRUVBOX["fg0"], va="bottom")

    for i, (name, hex_code) in enumerate(dark_accents):
        rect = patches.FancyBboxPatch(
            (i + 0.1, 0.42), 0.8, 0.58,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=hex_code, edgecolor=GRUVBOX["bg4"], linewidth=1.2
        )
        ax_dark.add_patch(rect)
        ax_dark.text(i + 0.5, 0.28, name, ha="center", va="top", color=GRUVBOX["fg1"], fontsize=9.5, fontweight="bold")
        ax_dark.text(i + 0.5, 0.06, hex_code, ha="center", va="top", color=GRUVBOX["fg4"], fontsize=8.5, fontfamily="monospace")

    # 3. Custom Colormaps: 3 colormaps side-by-side
    gradient = np.linspace(0, 1, 512).reshape(1, -1)

    cmaps = [
        (
            "gruvbox_rainbow",
            gruvbox_rainbow_cmap,
            "bg0 → blue_dark → aqua → green\n→ yellow → orange → red",
            gs[2, 0:2]
        ),
        (
            "gruvbox_heat",
            gruvbox_heat_cmap,
            "bg0 → red_dark → orange_dark\n→ orange → yellow → fg0",
            gs[2, 2:4]
        ),
        (
            "gruvbox_cool",
            gruvbox_cool_cmap,
            "bg0 → purple_dark → blue_dark\n→ blue → fg0",
            gs[2, 4:6]
        ),
    ]

    for title, cmap_obj, desc, grid_slice in cmaps:
        ax_c = fig.add_subplot(grid_slice)
        ax_c.text(0, 1.34, f"Colormap: {title}", fontsize=12, fontweight="bold", color=GRUVBOX["fg0"], va="bottom")
        ax_c.text(0, 1.06, desc, fontsize=7.5, color=GRUVBOX["fg4"], va="bottom", linespacing=1.2)
        ax_c.imshow(gradient, aspect="auto", cmap=cmap_obj, extent=[0, 1, 0, 1])
        ax_c.set_xlim(0, 1)
        ax_c.set_ylim(0, 1)
        ax_c.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_c.set_xticklabels(["0.0", "0.25", "0.50", "0.75", "1.00"], fontsize=8, color=GRUVBOX["fg3"])
        ax_c.set_yticks([])
        for spine in ax_c.spines.values():
            spine.set_color(GRUVBOX["bg4"])
            spine.set_linewidth(1.2)

    # 4. Backgrounds & Foregrounds
    ax_bg = fig.add_subplot(gs[3, 0:3])
    ax_bg.set_xlim(0, len(backgrounds))
    ax_bg.set_ylim(-0.1, 1.1)
    ax_bg.axis("off")
    ax_bg.text(0, 1.08, "Dark Background Tones", fontsize=11.5, fontweight="bold", color=GRUVBOX["fg0"], va="bottom")
    for i, (name, hex_code) in enumerate(backgrounds):
        rect = patches.FancyBboxPatch(
            (i + 0.1, 0.42), 0.8, 0.58,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=hex_code, edgecolor=GRUVBOX["bg4"], linewidth=1.0
        )
        ax_bg.add_patch(rect)
        ax_bg.text(i + 0.5, 0.28, name, ha="center", va="top", color=GRUVBOX["fg2"], fontsize=8.5, fontweight="bold")
        ax_bg.text(i + 0.5, 0.06, hex_code, ha="center", va="top", color=GRUVBOX["fg4"], fontsize=7.5, fontfamily="monospace")

    ax_fg = fig.add_subplot(gs[3, 3:6])
    ax_fg.set_xlim(0, len(foregrounds))
    ax_fg.set_ylim(-0.1, 1.1)
    ax_fg.axis("off")
    ax_fg.text(0, 1.08, "Light Foreground / Text Tones", fontsize=11.5, fontweight="bold", color=GRUVBOX["fg0"], va="bottom")
    for i, (name, hex_code) in enumerate(foregrounds):
        rect = patches.FancyBboxPatch(
            (i + 0.1, 0.42), 0.8, 0.58,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=hex_code, edgecolor=GRUVBOX["bg4"], linewidth=1.0
        )
        ax_fg.add_patch(rect)
        ax_fg.text(i + 0.5, 0.28, name, ha="center", va="top", color=GRUVBOX["fg2"], fontsize=8.5, fontweight="bold")
        ax_fg.text(i + 0.5, 0.06, hex_code, ha="center", va="top", color=GRUVBOX["fg4"], fontsize=7.5, fontfamily="monospace")

    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved {output_path} successfully.")

if __name__ == "__main__":
    create_gruvbox_palette_image("gruvbox.png")
