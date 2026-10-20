"""Compare GP CRPS rankings with embedded BIFT reduced-chi-squared results."""
import argparse
import csv
from html.parser import HTMLParser
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import rankdata, spearmanr

from plot_crps_quality import read_experiment
from run_gp_quality import read_simulation


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.in_cell = [], [], [], False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell, self.in_cell = [], True

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.in_cell:
            self.row.append("".join(self.cell).strip())
            self.in_cell = False
        elif tag == "tr" and self.row:
            self.rows.append(self.row)


def embedded_bift_rows(notebook):
    """Read the saved DataFrame HTML without rerunning the external BIFT binary."""
    document = json.loads(notebook.read_text())
    for cell in document["cells"]:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            html = "".join(html) if isinstance(html, list) else html
            if "SAXS_chi2_bift" not in html:
                continue
            parser = TableParser()
            parser.feed(html)
            header = parser.rows[0]
            result = []
            for values in parser.rows[1:]:
                if len(values) != len(header):
                    continue
                row = dict(zip(header, values))
                result.append({
                    "system": row["SYSTEM"],
                    "simulation_id": row["ID"],
                    "experiment": row["EXPERIMENT"],
                    "bift_chi2_reduced": float(row["SAXS_chi2_bift"]),
                })
            if result:
                return result
    raise ValueError(f"No embedded SAXS_chi2_bift table found in {notebook}")


def finite_lognorm(values):
    low, high = float(np.min(values)), float(np.max(values))
    low = max(low, np.finfo(float).tiny)
    if np.isclose(low, high):
        high = low*(1+1e-6)
    return LogNorm(low, high)


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=here/"outputs/databank_quality/gp_quality_scores.csv")
    parser.add_argument("--bift-notebook", type=Path,
                        default=root/"Scratch/saxs_bift_comparison_SO.ipynb")
    parser.add_argument("--output-dir", type=Path,
                        default=here/"outputs/databank_quality/crps_vs_bift")
    args = parser.parse_args()

    with args.scores.open(newline="") as handle:
        gp_rows = list(csv.DictReader(handle))
    bift = {(row["experiment"], row["simulation_id"]): row
            for row in embedded_bift_rows(args.bift_notebook)}
    matched = []
    for row in gp_rows:
        other = bift.get((row["experiment"], row["simulation_id"]))
        if other:
            matched.append({**row, **other})
    if not matched:
        raise ValueError("No matching experiment and simulation IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "system", "simulation_id", "trajectory",
              "gp_crps_sd_units", "bift_chi2_reduced", "gp_rank", "bift_rank",
              "absolute_rank_difference"]
    summaries = []
    for experiment in sorted({row["experiment"] for row in matched}):
        rows = [row for row in matched if row["experiment"] == experiment]
        crps = np.array([float(row["gp_crps_sd_units"]) for row in rows])
        chi2 = np.array([float(row["bift_chi2_reduced"]) for row in rows])
        crps_rank, chi2_rank = rankdata(crps), rankdata(chi2)
        for row, a, b in zip(rows, crps_rank, chi2_rank):
            row.update(gp_rank=float(a), bift_rank=float(b),
                       absolute_rank_difference=float(abs(a-b)))
        order = np.argsort(crps_rank)
        rows = [rows[index] for index in order]
        crps, chi2 = crps[order], chi2[order]
        crps_rank, chi2_rank = crps_rank[order], chi2_rank[order]
        rho, p_value = spearmanr(crps, chi2)
        summaries.append({"experiment": experiment, "matched_simulations": len(rows),
                          "spearman_rho": rho, "spearman_p": p_value,
                          "median_absolute_rank_difference": np.median(abs(crps_rank-chi2_rank)),
                          "maximum_absolute_rank_difference": np.max(abs(crps_rank-chi2_rank))})

        stem = experiment.replace("/", "_")
        with (args.output_dir/f"{stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)

        labels = [f'{row["system"]}:{row["simulation_id"]}' for row in rows]
        height = max(6, min(14, 2.4+.22*len(rows)))
        fig = plt.figure(figsize=(17, height), constrained_layout=True)
        grid = fig.add_gridspec(2, 4, width_ratios=(.75, .75, 2.7, .12),
                                height_ratios=(2.1, 1))
        crps_ax = fig.add_subplot(grid[:, 0]); chi_ax = fig.add_subplot(grid[:, 1])
        curve_ax = fig.add_subplot(grid[0, 2]); rank_ax = fig.add_subplot(grid[1, 2])
        cb_grid = grid[:, 3].subgridspec(2, 1, hspace=.35)
        crps_cax = fig.add_subplot(cb_grid[0]); chi_cax = fig.add_subplot(cb_grid[1])
        cmap = plt.get_cmap("RdYlGn_r")
        crps_norm, chi_norm = finite_lognorm(crps), finite_lognorm(chi2)
        crps_image = crps_ax.imshow(crps[:, None], cmap=cmap, norm=crps_norm,
                                    aspect="auto", origin="upper")
        chi_image = chi_ax.imshow(chi2[:, None], cmap=cmap, norm=chi_norm,
                                  aspect="auto", origin="upper")
        fontsize = 6 if len(rows) > 25 else 8
        crps_ax.set_yticks(np.arange(len(rows)), labels, fontsize=fontsize)
        chi_ax.set_yticks([])
        crps_ax.set_xticks([0], ["GP CRPS/SD"]); chi_ax.set_xticks([0], [r"BIFT $\chi_r^2$"])
        crps_ax.set_ylabel("Simulation (ordered by GP CRPS)")
        for ax, values, norm in ((crps_ax, crps, crps_norm), (chi_ax, chi2, chi_norm)):
            for index, value in enumerate(values):
                color = "white" if norm(value) > .62 else "black"
                ax.text(0, index, f"{value:.3g}", ha="center", va="center",
                        fontsize=fontsize, color=color)
        fig.colorbar(crps_image, cax=crps_cax).set_label("Integrated GP CRPS / mean predictive SD")
        fig.colorbar(chi_image, cax=chi_cax).set_label(r"BIFT reduced $\chi^2$")

        experiment_path = root/"Data/Experiments/saxs"/experiment/"saxs.dat"
        q_exp, intensity_exp, sigma_exp = read_experiment(experiment_path)
        use_exp = np.isfinite(intensity_exp) & (intensity_exp > 0)
        curve_ax.errorbar(q_exp[use_exp], intensity_exp[use_exp],
                          yerr=np.where(np.isfinite(sigma_exp[use_exp])
                                       & (sigma_exp[use_exp] > 0), sigma_exp[use_exp], 0),
                          fmt=".", markersize=2.5, linewidth=.5, color="black", alpha=.55,
                          zorder=5, label="Experiment")
        for row, score in zip(rows, crps):
            q_sim, intensity_sim = read_simulation(root/row["simulation_path"]/"SAXS.yaml")
            fitted = float(row["scale_sim_to_experiment"])*intensity_sim+float(row["background"])
            use = ((q_sim >= float(row["q_min_per_A"]))
                   & (q_sim <= float(row["q_max_per_A"])) & (fitted > 0))
            curve_ax.plot(q_sim[use], fitted[use], color=cmap(crps_norm(score)),
                          linewidth=1, alpha=.72)
        curve_ax.set(yscale="log", xlabel=r"$q$ (Å$^{-1}$)", ylabel=r"$I(q)$ (a.u.)",
                     title=f"{experiment}: matched simulations")
        curve_ax.grid(alpha=.18); curve_ax.legend(frameon=False)

        rank_ax.scatter(crps_rank, chi2_rank, c=crps, cmap=cmap, norm=crps_norm,
                        edgecolor="black", linewidth=.25)
        limit = len(rows)+1
        rank_ax.plot([0, limit], [0, limit], "--", color=".4", linewidth=1)
        disagreement = np.argsort(abs(crps_rank-chi2_rank))[-min(5, len(rows)):]
        for index in disagreement:
            rank_ax.annotate(labels[index], (crps_rank[index], chi2_rank[index]),
                             xytext=(3, 3), textcoords="offset points", fontsize=6)
        rank_ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="GP CRPS rank",
                    ylabel=r"BIFT $\chi_r^2$ rank",
                    title=fr"Rank agreement: Spearman $\rho$={rho:.3f}, $p$={p_value:.2g}")
        rank_ax.grid(alpha=.2)
        fig.savefig(args.output_dir/f"{stem}.png", dpi=450)
        plt.close(fig)

    with (args.output_dir/"rank_agreement_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    print(f"Wrote {len(matched)} matched comparisons across {len(summaries)} experiments")


if __name__ == "__main__":
    main()
