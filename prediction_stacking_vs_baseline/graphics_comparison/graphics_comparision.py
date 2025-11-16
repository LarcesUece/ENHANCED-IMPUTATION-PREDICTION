import os
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


data = {
    "link": ["sc-rn", "am-rn", "ac-rn", "pr-go", "mt-sc"],
    "model": ["Random Forest", "GRU", "LSTM", "GRU", "GRU"],
    "best_baseline_rmse": [0.0006847, 0.0567108, 0.0243952, 2.563495, 3.714655],
    "best_stacking_rmse": [0.0001815, 0.0283169, 0.0092732, 2.3236257, 3.34902544],
    "improvement_pct": [73.49, 50.06, 61.98, 9.39, 9.84]
}
df = pd.DataFrame(data)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_fig(fig, out_dir: str, filename: str, scale: int = 2):

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    try:
        fig.write_image(out_path, format="png", scale=scale)
        return out_path
    except ValueError as e:
        if "kaleido" in str(e).lower():
            fallback = out_path.rsplit('.', 1)[0] + '.html'
            fig.write_html(fallback, include_plotlyjs='cdn')
            print(
                f"[WARN] Kaleido package not available. Saved HTML instead: {fallback}\n"
                "Install kaleido inside your virtualenv to enable PNG export: pip install kaleido"
            )
            return fallback
        raise

def build_and_optionally_show(
    save_dir: str,
    show: bool,
    trend_color: str = "#E67E22",
    bar_color: str | None = None,
    bar_colors: list[str] | None = None,
    bar_palette: str | None = None,
    extra_model_charts: bool = False,
    bar_color_by_model: bool = False,
    model_colors_map: dict | None = None,
    improvement_bar_avg_line: bool = False,
    highlight_top_improvement: bool = False,
    improvement_bar_xmax: float | None = None,
):
    ordered = df.sort_values("improvement_pct", ascending=True)
    custom_color_list: list[str] | None = None

    if bar_palette:
        palette_name = bar_palette.strip()
        qualitative_palettes = {
            name: getattr(px.colors.qualitative, name)
            for name in dir(px.colors.qualitative)
            if not name.startswith("__") and isinstance(getattr(px.colors.qualitative, name), list)
        }
        if palette_name in qualitative_palettes:
            base_palette = qualitative_palettes[palette_name]
            custom_color_list = [base_palette[i % len(base_palette)] for i in range(len(ordered))]
        else:
            print(f"[WARN] Palette '{palette_name}' not found. Using default.")

    if bar_colors:
        cleaned = [c.strip() for c in bar_colors if c.strip()]
        if cleaned:
            custom_color_list = [cleaned[i % len(cleaned)] for i in range(len(ordered))]

    if bar_color:
        custom_color_list = [bar_color] * len(ordered)

    if bar_color_by_model and not (bar_color or bar_colors or bar_palette):
        fig_bar = px.bar(
            ordered,
            x="link",
            y="improvement_pct",
            orientation="v",
            color="model",
            text="improvement_pct",
            title="Improvement Percentage (Stacking vs Baseline) — by Model",
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    elif custom_color_list:
        fig_bar = go.Figure(
            go.Bar(
                x=ordered["improvement_pct"],
                y=ordered["link"],
                orientation="h",
                text=ordered["improvement_pct"],
                marker=dict(color=custom_color_list, line=dict(color="white", width=0.7)),
                hovertemplate="Link: %{y}<br>Improvement: %{x:.2f}%<extra></extra>",
            )
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    else:
        fig_bar = px.bar(
            ordered,
            x="improvement_pct",
            y="link",
            orientation="h",
            color="improvement_pct",
            color_continuous_scale="Viridis",
            text="improvement_pct",
            title="Improvement Percentage (Stacking vs Baseline)",
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig_bar.update_layout(
        title=fig_bar.layout.title.text or "Improvement Percentage (Stacking vs Baseline)",
        xaxis_title="Improvement (%)",
        yaxis_title="Link",
        plot_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=0.7, zeroline=False),
        margin=dict(l=70, r=40, t=60, b=40),
    )
    save_fig(fig_bar, save_dir, "improvement_bar.png")

    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        x=df["link"],
        y=df["best_baseline_rmse"],
        name="Baseline RMSE",
        marker_color="#4065C2"
    ))
    fig_rmse.add_trace(go.Bar(
        x=df["link"],
        y=df["best_stacking_rmse"],
        name="Stacking RMSE",
        marker_color="#27AE60"
    ))
    fig_rmse.update_layout(
        title="RMSE Comparison",
        barmode="group",
        yaxis_title="RMSE",
        xaxis_title="Link",
        plot_bgcolor="white",  
        legend_title="Imputation Type",
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=0.7, zeroline=False),
    )
    save_fig(fig_rmse, save_dir, "rmse_comparison.png")

    fig_bubble = px.scatter(
        df,
        x="link",
        y="improvement_pct",
        size="improvement_pct",
        color="model",
        hover_data=["best_baseline_rmse", "best_stacking_rmse"],
        size_max=60,
        title="Improvement Impact by Model (Bubble)",
    )
    fig_bubble.update_layout(
        xaxis_title="Link",
        yaxis_title="Improvement (%)",
        plot_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=0.7, zeroline=False),
    )
    save_fig(fig_bubble, save_dir, "improvement_bubble.png")

    fig_line = px.line(
        df,
        x="link",
        y="improvement_pct",
        markers=True,
        title="Improvement Percentage Stacking vs Baseline",
        text="improvement_pct",
    )
    fig_line.update_traces(
        line_color=trend_color,
        marker=dict(size=10, color=trend_color, line=dict(color="white", width=1)),
        texttemplate="%{text:.2f}%",
        textposition="top center",
        hovertemplate="Link: %{x}<br>Improvement: %{y:.2f}%<extra></extra>",
    )
    fig_line.update_layout(
        xaxis_title="Link",
        yaxis_title="Improvement (%)",
        plot_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=1.0, zeroline=False),
    )
    save_fig(fig_line, save_dir, "improvement_trend.png")

    fig_line_models = px.line( 
        df,
        x="link",
        y="improvement_pct",
        color="model",
        markers=True,
        title="Improvement Percentage Trend by Model",
        hover_data=["best_baseline_rmse", "best_stacking_rmse"],
    )
    fig_line_models.update_traces(line=dict(width=3))
    fig_line_models.update_layout(
        xaxis_title="Link",
        yaxis_title="Improvement (%)",
        plot_bgcolor="white",
        legend_title="Model",
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=1.0, zeroline=False),
    )
    save_fig(fig_line_models, save_dir, "improvement_trend_by_model.png")

    rmse_long = (
        df.melt(
            id_vars=["link"],
            value_vars=["best_baseline_rmse", "best_stacking_rmse"],
            var_name="rmse_type",
            value_name="rmse",
        )
        .replace({"best_baseline_rmse": "Baseline RMSE", "best_stacking_rmse": "Stacking RMSE"})
    )
    fig_rmse_lines = px.line(
        rmse_long,
        x="link",
        y="rmse",
        color="rmse_type",
        markers=True,
        title="RMSE of The Forecast - Trend",
        color_discrete_map={
            "Baseline RMSE": "#0C37A3", 
            "Stacking RMSE": "#0A9B46", 
        },
    )
    fig_rmse_lines.update_traces(line=dict(width=3))
    fig_rmse_lines.update_layout(
        xaxis_title="Link",
        yaxis_title="RMSE",
        plot_bgcolor="white",
        legend_title="Imputation Method",
        yaxis=dict(showgrid=True, gridcolor="#969292", gridwidth=0.7, zeroline=False),
    )
    save_fig(fig_rmse_lines, save_dir, "rmse_trend_lines.png")


    if extra_model_charts:
        ordered_desc = ordered.sort_values("improvement_pct", ascending=False)
        discrete_map = model_colors_map if model_colors_map else None
        fig_bar_model = px.bar(
            ordered_desc,
            x="improvement_pct",
            y="link",
            orientation="h",
            color="model",
            text="improvement_pct",
            title="Improvement (%) by Link and Model",
            color_discrete_map=discrete_map,
        )
        fig_bar_model.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        if improvement_bar_xmax is not None:
            xmax = improvement_bar_xmax
        else:
            xmax = ordered_desc["improvement_pct"].max() * 1.08 
        fig_bar_model.update_layout(
            xaxis_title="Improvement (%)",
            yaxis_title="Link",
            plot_bgcolor="white",
            yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=0.7, zeroline=False),
            xaxis=dict(range=[0, xmax]),
            margin=dict(l=70, r=70, t=60, b=40),  
            legend_title="Model",
        )
        if improvement_bar_avg_line:
            avg_imp = ordered_desc["improvement_pct"].mean()
            fig_bar_model.add_shape(
                type="line",
                x0=avg_imp,
                x1=avg_imp,
                y0=-0.5,
                y1=len(ordered_desc)-0.5,
                line=dict(color="#333333", width=2, dash="dash"),
            )
            fig_bar_model.add_annotation(
                x=avg_imp,
                y=-0.7,
                text=f"Avg: {avg_imp:.2f}%",
                showarrow=False,
                font=dict(size=12, color="#333333"),
                bgcolor="#F9F9F9",
                bordercolor="#333333",
                borderwidth=1,
                borderpad=3,
            )
        if highlight_top_improvement:
            top_row = ordered_desc.iloc[0]
            top_index = 0
            fig_bar_model.add_shape(
                type="rect",
                x0=0,
                x1=top_row["improvement_pct"],
                y0=top_index-0.4,
                y1=top_index+0.4,
                line=dict(color="#000000", width=1.5),
                fillcolor="rgba(255,255,0,0.10)",
            )
            fig_bar_model.add_annotation(
                x=top_row["improvement_pct"],
                y=top_index,
                text="TOP",
                showarrow=False,
                font=dict(color="#000000", size=11),
                bgcolor="rgba(255,255,0,0.25)",
            )
        save_fig(fig_bar_model, save_dir, "improvement_bar_by_model.png")

        fig_lollipop = go.Figure()
        for _, row in ordered.iterrows():
            fig_lollipop.add_trace(
                go.Scatter(
                    x=[0, row["improvement_pct"]],
                    y=[row["link"], row["link"]],
                    mode="lines+markers+text",
                    line=dict(width=4, color="#CCCCCC"),
                    marker=dict(
                        size=14,
                        color="#27AE60" if row["model"].lower().startswith("g") else "#4065C2",
                        line=dict(color="white", width=1),
                    ),
                    text=["", f"{row['improvement_pct']:.2f}%"],
                    textposition="top right",
                    name=row["model"],
                    hovertemplate=f"Link: {row['link']}<br>Model: {row['model']}<br>Improvement: {row['improvement_pct']:.2f}%<extra></extra>",
                    showlegend=False,
                )
            )
        fig_lollipop.update_layout(
            title="Improvement Lollipop Chart",
            xaxis_title="Improvement (%)",
            yaxis_title="Link",
            plot_bgcolor="white",
            yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=0.7, zeroline=False),
            xaxis=dict(showgrid=True, gridcolor="#E5E5E5", zeroline=False),
            margin=dict(l=70, r=40, t=60, b=40),
        )
        save_fig(fig_lollipop, save_dir, "improvement_lollipop.png")


        model_summary = (
            df.groupby("model", as_index=False)
              .agg(
                  mean_improvement=("improvement_pct", "mean"),
                  median_improvement=("improvement_pct", "median"),
                  links=("improvement_pct", "count"),
              )
        )
        fig_model_avg = px.bar(
            model_summary.sort_values("mean_improvement", ascending=False),
            x="mean_improvement",
            y="model",
            orientation="h",
            text="mean_improvement",
            color="model",
            title="Average Improvement (%) per Model",
        )
        fig_model_avg.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_model_avg.update_layout(
            xaxis_title="Mean Improvement (%)",
            yaxis_title="Model",
            plot_bgcolor="white",
            yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=0.7, zeroline=False),
            margin=dict(l=70, r=40, t=60, b=40),
        )
        save_fig(fig_model_avg, save_dir, "model_average_improvement.png")

    

    if show:
        figs = [fig_bar, fig_rmse, fig_bubble, fig_line, fig_line_models, fig_rmse_lines]
        if extra_model_charts:
            figs.extend([fig_bar_model, fig_lollipop, fig_model_avg])
        for fig in figs:
            fig.show()
            


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and optionally save improvement charts.")
    parser.add_argument("--out-dir", default="plots/summary_charts", help="Directory to save PNG charts")
    parser.add_argument("--show", action="store_true", help="Display interactive charts after saving")
    parser.add_argument("--trend-color", default="#E67E22", help="Hex color for improvement trend line (default: #E67E22)")
    parser.add_argument("--bar-color", help="Single hex color for all horizontal bars (e.g., #FF5733)")
    parser.add_argument("--bar-colors", help="Comma-separated list of colors for bars (e.g., #FF5733,#27AE60,#2980B9)")
    parser.add_argument("--bar-palette", help="Plotly qualitative palette name (e.g., Plotly, D3, Set1, Pastel)")
    parser.add_argument("--extra-model-charts", action="store_true", help="Generate extra charts highlighting models (lollipop, average per model, bar by model)")
    parser.add_argument("--bar-color-by-model", action="store_true", help="Color improvement bars by model instead of gradient or manual colors")
    parser.add_argument("--model-colors", help="Color map by model. Format: Model:Color,Model:Color (e.g., GRU:#4C78A8,LSTM:#F58518,Random Forest:#54A24B)")
    parser.add_argument("--improvement-bar-avg-line", action="store_true", help="Draw average line on the by-model chart")
    parser.add_argument("--highlight-top-improvement", action="store_true", help="Highlight the bar with highest improvement")
    parser.add_argument("--improvement-bar-xmax", type=float, help="Maximum X-axis value in the by-model chart (e.g., 75 to avoid cutting off 73.49% label)")
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    bar_colors_list = args.bar_colors.split(",") if getattr(args, "bar_colors", None) else None
    model_colors_map = None
    if args.model_colors:
        try:
            pairs = [p for p in args.model_colors.split(",") if p.strip()]
            mapping = {}
            for pair in pairs:
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    mapping[k.strip()] = v.strip()
            if mapping:
                model_colors_map = mapping
        except Exception as e:
            print(f"[WARN] Failed to parse --model-colors: {e}")
    build_and_optionally_show(
        args.out_dir,
        args.show,
        args.trend_color,
        bar_color=args.bar_color,
        bar_colors=bar_colors_list,
        bar_palette=args.bar_palette,
        extra_model_charts=args.extra_model_charts,
        bar_color_by_model=args.bar_color_by_model,
        model_colors_map=model_colors_map,
        improvement_bar_avg_line=args.improvement_bar_avg_line,
        highlight_top_improvement=args.highlight_top_improvement,
        improvement_bar_xmax=args.improvement_bar_xmax,
    )


