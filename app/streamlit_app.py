"""Interactive course-project demo backed by real NeuMF and GAP Rank artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from healthy_meal_recommender.demo import DemoData, DemoRecommendation, build_demo_view
from healthy_meal_recommender.demo import load_demo_data as load_project_demo_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
BANNER_PATH = APP_DIR / "assets" / "healthy-meal-banner.png"
STYLE_PATH = APP_DIR / "styles.css"
RESULT_PATH = PROJECT_ROOT / "results" / "reranking" / "gap_rank.json"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "mealrecplus_h"
REQUIRED_ARTIFACTS = (
    BANNER_PATH,
    STYLE_PATH,
    RESULT_PATH,
    PROCESSED_ROOT / "meal_catalog.csv",
    PROCESSED_ROOT / "user_health_train.csv",
    PROCESSED_ROOT / "candidates" / "neumf_standard_test_top100.csv",
)


st.set_page_config(
    page_title="Healthy Meal Navigator",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_data() -> DemoData:
    """Load immutable project outputs once per Streamlit process."""
    return load_project_demo_data(PROJECT_ROOT)


@st.cache_data
def load_offline_result() -> dict[str, object]:
    """Load the committed experiment result used by the evidence tab."""
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def representative_user(data: DemoData) -> int:
    """Choose the median positive-gap user for an informative default view."""
    records = sorted(
        (
            max(0.0, data.default_target - data.user_baselines[user_id]),
            user_id,
        )
        for user_id in data.candidates
        if data.user_baselines[user_id] < data.default_target
    )
    return records[len(records) // 2][1]


def recommendation_table(rows: tuple[DemoRecommendation, ...]) -> pd.DataFrame:
    """Create the compact table shown in the main task surface."""
    return pd.DataFrame(
        [
            {
                "Rank": row.rank,
                "Change": "Promoted" if row.promoted else "Kept",
                "Meal ID": row.candidate.meal.meal_id,
                "Appetizer": row.candidate.meal.courses[0],
                "Main dish": row.candidate.meal.courses[1],
                "Dessert": row.candidate.meal.courses[2],
                "Health": row.candidate.meal.health_utility,
                "Preference": row.candidate.preference_score,
                "Goal": "Meets" if row.meets_target else "Below",
            }
            for row in rows
        ]
    )


def comparison_chart(view) -> go.Figure:
    """Plot health utility by rank before and after reranking."""
    ranks = list(range(1, 11))
    before = [row.candidate.meal.health_utility for row in view.before]
    after = [row.candidate.meal.health_utility for row in view.after]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=ranks,
            y=before,
            mode="lines+markers",
            name="Preference only",
            line={"color": "#829AB1", "width": 2},
            marker={"size": 7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ranks,
            y=after,
            mode="lines+markers",
            name="GAP Rank",
            line={"color": "#0F766E", "width": 3},
            marker={"size": 8},
        )
    )
    figure.add_hline(
        y=view.target,
        line_dash="dash",
        line_color="#C75A18",
        annotation_text="Selected target",
        annotation_position="top left",
    )
    figure.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis={"title": "Recommendation rank", "dtick": 1},
        yaxis={"title": "Health utility", "range": [0.1, 0.6]},
        legend={"orientation": "h", "y": 1.13, "x": 0},
        hovermode="x unified",
    )
    return figure


def render_meal_details(rows: tuple[DemoRecommendation, ...]) -> None:
    """Show nutrition and source-model details for one selected recommendation."""
    options = {f"#{row.rank} · Meal {row.candidate.meal.meal_id}": row for row in rows}
    selected_label = st.selectbox("Inspect a recommended meal", options)
    selected = options[selected_label]
    meal = selected.candidate.meal
    st.markdown(f"### {meal.courses[1]}")
    st.caption(
        f"Appetizer: {meal.courses[0]}  ·  Dessert: {meal.courses[2]}"
    )
    detail_columns = st.columns(4)
    nutrition = (
        ("Calories", meal.calories_kcal, "kcal"),
        ("Protein", meal.protein_g, "g"),
        ("Carbohydrates", meal.carbohydrates_g, "g"),
        ("Sodium", meal.sodium_mg, "mg"),
    )
    for column, (label, value, unit) in zip(detail_columns, nutrition, strict=True):
        display = "Unavailable" if value is None else f"{value:,.1f} {unit}"
        column.metric(label, display)
    st.caption(
        f"FSA {meal.fsa_score:.2f} (lower is healthier) · "
        f"WHO {meal.who_score:.2f} (higher is healthier) · "
        f"NeuMF score {selected.candidate.preference_score:.3f} is a ranking score, "
        "not a probability."
    )


def render_method_cards() -> None:
    """Explain the live method without unnecessary mathematical detail."""
    columns = st.columns(3)
    cards = (
        (
            "1 · Measure the gap",
            "The user baseline comes only from training interactions. If it already "
            "meets the selected target, GAP Rank leaves the preference list unchanged.",
        ),
        (
            "2 · Check feasibility",
            "The method checks the real NeuMF Top-100. A target quota never exceeds the "
            "number of target-meeting candidates that actually exist.",
        ),
        (
            "3 · Make minimal changes",
            "Only the highest-preference target meals needed to fill the quota are "
            "promoted. Other retained meals keep their preference order.",
        ),
    )
    for column, (title, text) in zip(columns, cards, strict=True):
        column.markdown(
            f'<div class="method-card"><strong>{title}</strong><span>{text}</span></div>',
            unsafe_allow_html=True,
        )


def render_offline_evidence(result: dict[str, object]) -> None:
    """Show the held-out comparison directly from the committed result JSON."""
    labels = {
        "preference_only": "Preference only",
        "fixed_weight": "Fixed weight",
        "hard_filter": "Hard filter",
        "gap_rank": "GAP Rank",
    }
    rows = []
    for method in ("preference_only", "fixed_weight", "hard_filter", "gap_rank"):
        metrics = result["test"][method]
        rows.append(
            {
                "Method": labels[method],
                "NDCG@10": metrics["ndcg@10"],
                "NDCG retained": metrics["ndcg@10_retention"],
                "Health utility": metrics["mean_health_utility@10"],
                "Goal item rate": metrics["goal_item_rate@10"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "NDCG@10": st.column_config.NumberColumn(format="%.4f"),
            "NDCG retained": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.1%%"
            ),
            "Health utility": st.column_config.NumberColumn(format="%.3f"),
            "Goal item rate": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.1%%"
            ),
        },
    )


missing = [path for path in REQUIRED_ARTIFACTS if not path.is_file()]
if missing:
    st.error("The reproducible demo artifacts are not ready on this machine.")
    st.code(
        ".\\.venv\\Scripts\\python.exe .\\scripts\\prepare_mealrecplus.py\n"
        ".\\.venv\\Scripts\\python.exe .\\scripts\\run_neumf.py\n"
        ".\\.venv\\Scripts\\python.exe .\\scripts\\export_candidates.py\n"
        ".\\.venv\\Scripts\\python.exe .\\scripts\\run_reranking.py",
        language="powershell",
    )
    st.caption("Missing: " + ", ".join(path.name for path in missing))
    st.stop()

st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
data = load_data()
offline_result = load_offline_result()

hero_copy, hero_image = st.columns([1.55, 1], gap="large", vertical_alignment="center")
with hero_copy:
    st.markdown('<div class="eyebrow">CA6124 · Personalized Wellness</div>', unsafe_allow_html=True)
    st.title("Healthy Meal Navigator")
    st.markdown(
        '<div class="hero-copy">Explore how GAP Rank turns real NeuMF candidates into '
        "a healthier Top-10 while keeping each anonymous user's learned preferences visible.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-row">'
        '<span class="status-pill">1,114 anonymous demo users</span>'
        '<span class="status-pill">Real Top-100 candidates</span>'
        '<span class="status-pill">No patient data</span>'
        "</div>",
        unsafe_allow_html=True,
    )
with hero_image:
    st.image(BANNER_PATH, width="stretch")

st.markdown("## Personalize the recommendation")
st.markdown(
    '<div class="section-note">Choose an anonymous benchmark user and a wellness target. '
    "The recommendation is recalculated immediately from that user's NeuMF candidates.</div>",
    unsafe_allow_html=True,
)
users = sorted(data.candidates)
default_user = representative_user(data)
controls = st.container(border=True)
user_column, target_column = controls.columns([1, 1.35], gap="large")
with user_column:
    selected_user = st.selectbox(
        "Anonymous user",
        users,
        index=users.index(default_user),
        format_func=lambda user_id: (
            f"User {user_id:04d} · baseline {data.user_baselines[user_id]:.3f}"
        ),
        help="Numeric identifiers are anonymized dataset indexes, not personal identities.",
    )
with target_column:
    selected_target = st.slider(
        "Wellness target",
        min_value=0.20,
        max_value=0.55,
        value=round(data.default_target, 3),
        step=0.005,
        format="%.3f",
        help="Higher values request more target-meeting meals. This is an offline ranking target, not a clinical threshold.",
    )

view = build_demo_view(data, selected_user, selected_target)
if view.gap == 0.0:
    st.info(
        "This user's training baseline already meets the selected target, so GAP Rank "
        "preserves the original Top-10. Raise the target to explore an intervention."
    )
elif view.target_quota == 0:
    st.warning(
        "The selected target cannot be improved with this user's current Top-100, so the "
        "preference order is preserved."
    )

profile_metrics = st.columns(3)
profile_metrics[0].metric("Training baseline", f"{view.baseline:.3f}")
profile_metrics[1].metric(
    "Selected target", f"{view.target:.3f}", f"gap {view.gap:.3f}"
)
profile_metrics[2].metric("Adaptive quota", f"{view.target_quota} / 10")
outcome_metrics = st.columns(3)
outcome_metrics[0].metric(
    "Top-10 health",
    f"{view.after_mean_health:.3f}",
    f"{view.after_mean_health - view.before_mean_health:+.3f} vs before",
)
outcome_metrics[1].metric(
    "Goal coverage",
    f"{view.after_goal_rate:.0%}",
    f"{view.after_goal_rate - view.before_goal_rate:+.0%} vs before",
)
outcome_metrics[2].metric("Preference overlap", f"{view.top10_overlap:.0%}")

recommendations_tab, comparison_tab, method_tab = st.tabs(
    ["Recommendations", "Before & after", "Method & evidence"]
)
with recommendations_tab:
    st.markdown("### GAP Rank Top-10")
    st.caption(
        "Promoted means the meal entered the Top-10 to satisfy this user's adaptive quota."
    )
    st.dataframe(
        recommendation_table(view.after),
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small", format="%d"),
            "Change": st.column_config.TextColumn(width="small"),
            "Meal ID": st.column_config.NumberColumn(width="small", format="%d"),
            "Health": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Preference": st.column_config.NumberColumn(format="%.3f"),
            "Goal": st.column_config.TextColumn(width="small"),
        },
    )
    st.divider()
    render_meal_details(view.after)

with comparison_tab:
    st.markdown("### Health utility across recommendation ranks")
    st.caption(
        "The model promotes only enough target-meeting candidates to fill the adaptive quota."
    )
    st.plotly_chart(comparison_chart(view), width="stretch", config={"displayModeBar": False})
    before_column, after_column = st.columns(2, gap="large")
    with before_column:
        st.markdown("#### Preference only")
        st.dataframe(
            recommendation_table(view.before)[["Rank", "Meal ID", "Main dish", "Health", "Goal"]],
            hide_index=True,
            width="stretch",
        )
    with after_column:
        st.markdown("#### GAP Rank")
        st.dataframe(
            recommendation_table(view.after)[
                ["Rank", "Change", "Meal ID", "Main dish", "Health", "Goal"]
            ],
            hide_index=True,
            width="stretch",
        )

with method_tab:
    st.markdown("### One adaptive rule, three checks")
    render_method_cards()
    st.markdown("### Held-out test evidence")
    st.caption(
        "Loaded from the committed experiment result. Parameters were selected on validation data."
    )
    render_offline_evidence(offline_result)
    st.markdown(
        '<div class="safety-note"><strong>Wellness boundary.</strong> The health utility is '
        "an interpretable combination of dataset-provided FSA and WHO scores. It is not a "
        "diagnosis, prescription, or guarantee of health outcomes.</div>",
        unsafe_allow_html=True,
    )
