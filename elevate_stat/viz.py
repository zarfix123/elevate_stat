import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import networkx as nx

ACCENT = "#4fc3f7"


def centrality_bar(centrality_df, path, names=None, n=20, title="LATE — Top Teammate Elevators"):
    d = centrality_df.head(n).iloc[::-1]
    labels = d["PLAYER_ID"].map(names) if names else d["PLAYER_ID"].astype(str)
    fig, ax = plt.subplots(figsize=(9, n * 0.38 + 1))
    ax.barh(range(len(d)), d["elevation_centrality"], color=ACCENT)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Elevation centrality  (teammate pts / 100 poss above baseline)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def who_lifts_whom(pairs_df, players, names, path, min_lift=0.0):
    players = list(players)
    sub = pairs_df[pairs_df["A"].isin(players) & pairs_df["B"].isin(players)
                   & (pairs_df["lift"] > min_lift)]
    G = nx.DiGraph()
    G.add_nodes_from(players)
    for r in sub.itertuples(index=False):
        G.add_edge(r.A, r.B, weight=r.lift)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=1, k=1.3)
    widths = [G[u][v]["weight"] for u, v in G.edges()]
    maxw = max(widths) if widths else 1.0
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ACCENT, node_size=700, edgecolors="#263238")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#546e7a", alpha=0.55, arrowsize=14,
                           width=[1 + 4 * w / maxw for w in widths],
                           connectionstyle="arc3,rad=0.08", node_size=700)
    label_pos = {p: (x, y - 0.11) for p, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=10, font_color="black",
                            labels={p: names.get(p, str(p)) for p in players},
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.3))
    ax.set_title("Who lifts whom  (arrow A→B: A makes B more efficient; thicker = bigger lift)")
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    return path


def clutch_scatter(ratings_df, path, names=None, min_minutes=12000, n_annot=8):
    d = ratings_df[ratings_df["minutes"] >= min_minutes].copy()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(d["rapm"], d["late"], s=18, color=ACCENT, alpha=0.7)
    lim = [min(d["rapm"].min(), d["late"].min()) - 1, max(d["rapm"].max(), d["late"].max()) + 1]
    ax.plot(lim, lim, color="gray", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    d["rise"] = d["late"] - d["rapm"]
    for r in d.sort_values("rise", ascending=False).head(n_annot).itertuples(index=False):
        nm = names.get(r.PLAYER_ID, "") if names else str(r.PLAYER_ID)
        ax.annotate(nm, (r.rapm, r.late), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("RAPM  (all possessions)")
    ax.set_ylabel("LATE  (leverage-weighted)")
    ax.set_title("Who rises in high-leverage minutes  (above the line = bigger when it matters)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def mechanism_map(mech_df, path, names=None, n=18):
    d = mech_df.copy()
    d["total"] = d["vol_centrality"] + d["eff_centrality"]
    d = d.sort_values("total", ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(d["vol_centrality"], d["eff_centrality"], s=45, color=ACCENT)
    for r in d.itertuples(index=False):
        nm = names.get(r.PLAYER_ID, "") if names else str(r.PLAYER_ID)
        ax.annotate(nm, (r.vol_centrality, r.eff_centrality), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("Volume channel  →  creates shots for teammates")
    ax.set_ylabel("Efficiency channel  →  improves teammate shot quality")
    ax.set_title("How players elevate teammates: creation vs. shot-quality")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def trajectory(metrics_df, player_ids, names, path):
    d = metrics_df[metrics_df["season_type"] == "Regular Season"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for pid in player_ids:
        pdf = d[d["PLAYER_ID"] == pid].sort_values("season")
        if not pdf.empty:
            ax.plot(pdf["season"], pdf["points_above_expected"], marker="o",
                    label=names.get(pid, str(pid)))
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("Shot-making: points above expected (xPPS)")
    ax.set_title("Career shot-making trajectory")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def archetype_bar(arch_df, player_id, name, path):
    d = arch_df[arch_df["A"] == player_id].sort_values("archetype")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["archetype"].astype(int).astype(str), d["lift"], color=ACCENT)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Teammate archetype")
    ax.set_ylabel("Lift (pts / 100 poss)")
    ax.set_title(f"{name}: elevation by teammate archetype")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
