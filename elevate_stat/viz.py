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
