import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="EuroLeague 24/25 — Mini App", layout="wide")

DATA_PATH_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "euroleague_24_25_dummy.csv"

@st.cache_data
def load_data(from_upload=None):
    import re
    def _read(pathlike):
        df = pd.read_csv(pathlike)
        # normalizuj nazive kolona: lowercase, zameni razmake i % → _pct
        new_cols = []
        for c in df.columns:
            c_norm = c.strip().lower()
            c_norm = c_norm.replace("%", "_pct")
            c_norm = re.sub(r"[^a-z0-9_]+", "_", c_norm)
            new_cols.append(c_norm)
        df.columns = new_cols

        # date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "datum" in df.columns:
            df["date"] = pd.to_datetime(df["datum"], errors="coerce")
        else:
            # ako nema datuma, napravi veštački po redu
            df["date"] = pd.to_datetime(range(len(df)), unit="D", origin="2024-10-01")

        # team / opponent
        if "team" not in df.columns:
            for alt in ["club", "home_team"]:
                if alt in df.columns:
                    df["team"] = df[alt]
                    break
        if "opponent" not in df.columns:
            for alt in ["away_team", "opp"]:
                if alt in df.columns:
                    df["opponent"] = df[alt]
                    break

        # result / win
        if "result" not in df.columns:
            if "win" in df.columns:
                df["win"] = df["win"].astype(int)
                df["result"] = df["win"].map({1: "W", 0: "L"})
            else:
                df["result"] = "W"
                df["win"] = 1
        else:
            df["win"] = (df["result"].astype(str).str.upper() == "W").astype(int)

        # points
        if "points_for" not in df.columns:
            if "points" in df.columns:
                df["points_for"] = df["points"]
            elif "pf" in df.columns:
                df["points_for"] = df["pf"]

        if "points_against" not in df.columns and "pa" in df.columns:
            df["points_against"] = df["pa"]

        # shooting %
        if "fg_pct" not in df.columns:
            if "fg" in df.columns:
                df["fg_pct"] = df["fg"]
            elif {"fgm","fga"} <= set(df.columns):
                df["fg_pct"] = np.where(df["fga"] > 0, df["fgm"] / df["fga"], np.nan)
        if "fg3_pct" not in df.columns:
            if "3p_pct" in df.columns:
                df["fg3_pct"] = df["3p_pct"]
            elif {"fg3m","fg3a"} <= set(df.columns):
                df["fg3_pct"] = np.where(df["fg3a"] > 0, df["fg3m"] / df["fg3a"], np.nan)
        if "ft_pct" not in df.columns:
            if {"ftm","fta"} <= set(df.columns):
                df["ft_pct"] = np.where(df["fta"] > 0, df["ftm"] / df["fta"], np.nan)

        # point diff
        if "points_against" in df.columns:
            df["point_diff"] = df["points_for"] - df["points_against"]
        else:
            df["point_diff"] = np.nan

        # home/away fallback
        if "home_away" not in df.columns:
            df["home_away"] = "H"

        return df

    if from_upload is not None:
        return _read(from_upload)
    else:
        return _read(DATA_PATH_DEFAULT)


def _mean_safe(df, col, default=np.nan):
    return df[col].mean() if col in df.columns and not df.empty else default

def kpi_cards(df_team):
    gp = len(df_team)
    wins = int(df_team["win"].sum()) if "win" in df_team.columns else 0
    win_pct = (wins / gp) * 100 if gp > 0 else 0.0

    ppg  = _mean_safe(df_team, "points_for", 0.0)
    oppg = _mean_safe(df_team, "points_against", np.nan)

    diff = (ppg - oppg) if not np.isnan(oppg) else np.nan
    fg   = _mean_safe(df_team, "fg_pct", np.nan)
    fg3  = _mean_safe(df_team, "fg3_pct", np.nan)
    ast  = _mean_safe(df_team, "ast", np.nan)
    tov  = _mean_safe(df_team, "tov", np.nan)
    ast_to = (ast / tov) if (not np.isnan(ast) and not np.isnan(tov) and tov != 0) else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", gp)
    c2.metric("Wins", wins, f"{win_pct:.1f}%")

    if np.isnan(oppg):
        c3.metric("PPG", f"{ppg:.1f}")
    else:
        c3.metric("PPG / OPPG", f"{ppg:.1f} / {oppg:.1f}", f"Δ {diff:+.1f}")

    if not np.isnan(fg) and not np.isnan(fg3):
        c4.metric("FG% / 3P%", f"{(fg*100):.1f}% / {(fg3*100):.1f}%")
    else:
        c4.metric("FG% / 3P%", "—")

    c5.metric("AST/TO", f"{ast_to:.2f}" if not np.isnan(ast_to) else "—")


def league_table(df):
    agg_dict = {}
    if "date" in df.columns: agg_dict["date"] = "count"
    if "win" in df.columns: agg_dict["win"] = "sum"
    if "points_for" in df.columns: agg_dict["points_for"] = "mean"
    if "points_against" in df.columns: agg_dict["points_against"] = "mean"
    if "fg_pct" in df.columns: agg_dict["fg_pct"] = "mean"
    if "fg3_pct" in df.columns: agg_dict["fg3_pct"] = "mean"

    g = df.groupby("team").agg(agg_dict).reset_index()

    if "date" in g.columns: g = g.rename(columns={"date": "GP"})
    if "win" in g.columns: g = g.rename(columns={"win": "W"})
    if "points_for" in g.columns: g = g.rename(columns={"points_for": "PF"})
    if "points_against" in g.columns: g = g.rename(columns={"points_against": "PA"})
    if "fg_pct" in g.columns: g = g.rename(columns={"fg_pct": "FG%"})
    if "fg3_pct" in g.columns: g = g.rename(columns={"fg3_pct": "3P%"})

    if "GP" in g.columns and "W" in g.columns:
        g["L"] = g["GP"] - g["W"]
        g["WIN%"] = (g["W"] / g["GP"]) * 100
    else:
        g["L"] = np.nan
        g["WIN%"] = np.nan

    if "point_diff" in df.columns:
        team_diff = df.groupby("team")["point_diff"].mean().rename("DIFF").reset_index()
        g = g.merge(team_diff, on="team", how="left")
    else:
        g["DIFF"] = np.nan

    sort_cols = [c for c in ["WIN%", "DIFF"] if c in g.columns]
    if sort_cols:
        g = g.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    for c in ["PF", "PA"]:
        if c in g.columns: g[c] = g[c].round(1)
    if "FG%" in g.columns: g["FG%"] = (g["FG%"] * 100).round(1)
    if "3P%" in g.columns: g["3P%"] = (g["3P%"] * 100).round(1)

    keep = [c for c in ["team","GP","W","L","WIN%","PF","PA","DIFF","FG%","3P%"] if c in g.columns]
    return g[keep]


def team_trend(df_team):
    df_team = df_team.sort_values(df_team.columns.intersection(["date"]).tolist()).copy()
    if "point_diff" not in df_team.columns:
        if {"points_for","points_against"} <= set(df_team.columns):
            df_team["point_diff"] = df_team["points_for"] - df_team["points_against"]
        else:
            df_team["point_diff"] = df_team.get("points_for", pd.Series([np.nan]*len(df_team)))
    df_team["rolling_diff"] = df_team["point_diff"].rolling(3, min_periods=1).mean()
    index_col = "date" if "date" in df_team.columns else None
    if index_col:
        st.line_chart(df_team.set_index(index_col)[["point_diff","rolling_diff"]], height=260)
    else:
        st.line_chart(df_team[["point_diff","rolling_diff"]], height=260)


def head_to_head(df, team, opponent=None):
    sub = df[df.get("team") == team].copy() if "team" in df.columns else df.copy()
    if opponent and opponent != "All opponents" and "opponent" in sub.columns:
        sub = sub[sub["opponent"] == opponent]

    sort_cols = [c for c in ["date", "round"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols)

    preferred = ["round","date","home_away","opponent","points_for","points_against",
                 "result","fgm","fga","fg3m","fg3a","ftm","fta","reb","ast","tov"]
    cols = [c for c in preferred if c in sub.columns]
    if cols:
        st.dataframe(sub[cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(sub, use_container_width=True, hide_index=True)


# ==== Sidebar ====
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = load_data(uploaded)

# Guard: ako nema 'team', napravi placeholder
if "team" not in df.columns:
    st.error("U datasetu ne postoji kolona 'team'. Dodaj 'team' ili mapiraj naziv kluba na 'team'.")
    st.stop()

teams = sorted(df["team"].dropna().unique().tolist())
team = st.sidebar.selectbox("Team", options=teams, index=0)

view_options = ["Overview", "League Table", "Trends", "Head-to-Head"]
view = st.sidebar.radio("View", view_options, index=0)

# ==== Main ====
st.title("🏀 EuroLeague 24/25 — Mini App")
st.caption("Korak 4: Interaktivna aplikacija")

df_team = df[df["team"] == team]

if view == "Overview":
    st.subheader(f"🔎 {team} — Overview")
    kpi_cards(df_team)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Shooting Snapshot")
        # koristi ono što postoji: ili attempts/makes, ili procente
        shot_cols = [c for c in ["fgm","fga","fg3m","fg3a","ftm","fta"] if c in df_team.columns]
        if shot_cols:
            snap = df_team[shot_cols].mean(numeric_only=True).round(1)
            st.bar_chart(snap, height=260)
        else:
            pct_map = [("FG%", "fg_pct"), ("3P%", "fg3_pct"), ("FT%", "ft_pct")]
            avail = {name: (df_team[col].mean() * 100)
                     for name, col in pct_map if col in df_team.columns}
            if len(avail) > 0:
                snap = pd.Series(avail).round(1)
                st.bar_chart(snap, height=260)
            else:
                st.info("Nema dostupnih kolona za šut (FG/3P/FT).")

    with col2:
        st.markdown("#### REB / AST / TOV (avg)")
        misc_cols = [c for c in ["reb","ast","tov"] if c in df_team.columns]
        if misc_cols:
            misc = df_team[misc_cols].mean(numeric_only=True)
            rename_map = {"reb":"REB","ast":"AST","tov":"TOV"}
            misc.index = [rename_map.get(i, i.upper()) for i in misc.index]
            st.bar_chart(misc, height=260)
        else:
            st.info("Nema REB/AST/TOV kolona u datasetu.")

    st.markdown("#### Last games")
    head_to_head(df, team, opponent=None)

elif view == "League Table":
    st.subheader("📊 League Table")
    st.dataframe(league_table(df), use_container_width=True, hide_index=True)

elif view == "Trends":
    st.subheader(f"📈 {team} — Trends")
    team_trend(df_team)

elif view == "Head-to-Head":
    st.subheader(f"🤝 {team} — Head-to-Head")
    if "opponent" in df.columns:
        opps = ["All opponents"] + sorted(df["opponent"].dropna().unique().tolist())
    else:
        opps = ["All opponents"]
    opp_choice = st.selectbox("Opponent", options=opps, index=0)
    head_to_head(df, team, opponent=opp_choice)
