# ============================================================
# STREAMLIT INTERACTIVE DASHBOARD
# File: app.py
# Run with: streamlit run app.py
# Install:  pip install streamlit pandas matplotlib seaborn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Screen Time EDA",
    page_icon="📱",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #e91e8c, #3498db);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .metric-box {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center; border: 1px solid #e0e0e0;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #e91e8c; }
    .metric-label { font-size: 0.8rem; color: #666; }
</style>
""", unsafe_allow_html=True)


# ── Load & Clean Data ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('students_screentime.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    for col in ['Environment', 'Work Strategy', 'Notification Handling']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    def simplify_app(x):
        if 'Social Media' in str(x): return 'Social Media'
        if 'Productivity' in str(x): return 'Productivity'
        if 'Streaming'    in str(x): return 'Streaming'
        if 'Messaging'    in str(x): return 'Messaging'
        if 'Gaming'       in str(x): return 'Gaming'
        return 'Other'

    df['App_Category'] = df['App Category'].apply(simplify_app)
    df['Att_Score']    = df['Attention Span'].map({
        'Less than 10 minutes': 1, '10–30 minutes': 2,
        '30–60 minutes': 3, 'More than 1 hour': 4
    })
    df['Screen_Hours'] = df['Average Screen Time'].map({
        'Less than 2': 1, '2–4': 3, '4–6': 5,
        '6–8': 7, '8-10': 9, 'More than 10': 11
    })
    df['Prod_Score'] = df['Productivity'].map({
        'Unproductive, i might not have completed the task and got carried away': 1,
        'Moderately productive': 2,
        'Extremely productive, i efficiently complete my tasks': 3
    })
    return df

df = load_data()

screen_order = ['Less than 2', '2–4', '4–6', '6–8', '8-10', 'More than 10']
att_order    = ['Less than 10 minutes', '10–30 minutes', '30–60 minutes', 'More than 1 hour']
period_order = ['Morning (6 AM–12 PM)', 'Afternoon (12 PM–6 PM)',
                'Evening (6 PM–10 PM)', 'Late night (10 PM–6 AM)']


# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-title">📱 Student Screen Time EDA Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Batch F-13 | 50,000 respondents | Exploratory Data Analysis</p>',
            unsafe_allow_html=True)
st.divider()


# ── Sidebar Filters ──────────────────────────────────────────
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("Filter the dataset across all charts")

sel_gender = st.sidebar.multiselect(
    "Gender", options=df['Gender'].unique().tolist(),
    default=df['Gender'].unique().tolist()
)
sel_occ = st.sidebar.multiselect(
    "Occupation", options=df['Occupation'].unique().tolist(),
    default=df['Occupation'].unique().tolist()
)
sel_age = st.sidebar.multiselect(
    "Age Group", options=['Below 18','18–24','25–34','35–44','45 and above'],
    default=['Below 18','18–24','25–34','35–44','45 and above']
)
sel_app = st.sidebar.multiselect(
    "App Category", options=df['App_Category'].unique().tolist(),
    default=df['App_Category'].unique().tolist()
)

# Apply filters
fdf = df[
    df['Gender'].isin(sel_gender) &
    df['Occupation'].isin(sel_occ) &
    df['Age Group'].isin(sel_age) &
    df['App_Category'].isin(sel_app)
]

if fdf.empty:
    st.warning("No data matches your filters. Please adjust the sidebar filters.")
    st.stop()

st.sidebar.markdown(f"**Showing:** {len(fdf):,} / {len(df):,} records")


# ── KPI Metrics ───────────────────────────────────────────────
st.subheader("📊 Key Metrics")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Total Records",    f"{len(fdf):,}")
with c2:
    st.metric("Avg Attention Score", f"{fdf['Att_Score'].mean():.2f} / 4")
with c3:
    st.metric("Avg Screen Time",  f"{fdf['Screen_Hours'].mean():.1f} hrs/day")
with c4:
    st.metric("Avg Productivity", f"{fdf['Prod_Score'].mean():.2f} / 3")
with c5:
    top_app = fdf['App_Category'].mode()[0]
    st.metric("Most Used App",    top_app)

st.divider()


# ── Row 1: Distributions ─────────────────────────────────────
st.subheader("📈 Distributions")
r1c1, r1c2 = st.columns(2)

with r1c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = fdf['Attention Span'].value_counts().reindex(att_order)
    colors = ['#e74c3c','#e67e22','#3498db','#2ecc71']
    ax.bar(att_order, counts.values, color=colors, edgecolor='white')
    ax.set_title('Attention Span Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=20)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f'{v:,}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with r1c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    st_counts = fdf['Average Screen Time'].value_counts().reindex(screen_order).dropna()
    ax.bar(st_counts.index, st_counts.values, color='#3498db', edgecolor='white')
    ax.set_title('Screen Time Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Row 2: Core Analysis ──────────────────────────────────────
st.subheader("🔗 Screen Time vs Attention Span")
r2c1, r2c2 = st.columns(2)

with r2c1:
    avg_att = fdf.groupby('Average Screen Time')['Att_Score'].mean().reindex(screen_order).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(avg_att.index, avg_att.values,
                  color=['#2ecc71','#3498db','#9b59b6','#e67e22','#e74c3c','#c0392b'],
                  edgecolor='white')
    ax.axhline(fdf['Att_Score'].mean(), color='gray', linestyle='--', linewidth=1.2, label='Avg')
    ax.set_title('Avg Attention Score by Screen Time', fontweight='bold')
    ax.set_ylabel('Avg Attention Score (1–4)')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    for bar, val in zip(bars, avg_att.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with r2c2:
    app_att = fdf.groupby('App_Category')['Att_Score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_val = fdf['Att_Score'].mean()
    bar_colors = ['#2ecc71' if v >= mean_val else '#e74c3c' for v in app_att.values]
    ax.bar(app_att.index, app_att.values, color=bar_colors, edgecolor='white')
    ax.axhline(mean_val, color='gray', linestyle='--', linewidth=1.2, label='Avg')
    ax.set_title('Attention Score by App Category', fontweight='bold')
    ax.set_ylabel('Avg Attention Score (1–4)')
    ax.legend()
    for i, (idx, val) in enumerate(app_att.items()):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Row 3: Peak Distraction & Heatmap ────────────────────────
st.subheader("⏰ Peak Distraction Analysis")
r3c1, r3c2 = st.columns(2)

with r3c1:
    low_att = fdf[fdf['Attention Span'] == 'Less than 10 minutes']
    dist_counts = low_att['Screen Time Period'].value_counts().reindex(period_order).fillna(0)
    period_labels = ['Morning\n6AM–12PM', 'Afternoon\n12PM–6PM',
                     'Evening\n6PM–10PM', 'Late Night\n10PM–6AM']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(period_labels, dist_counts.values,
                  color=['#3498db','#f39c12','#e74c3c','#9b59b6'], edgecolor='white')
    ax.set_title('Peak Distraction Hours\n(Attention < 10 min)', fontweight='bold')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, dist_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(val):,}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with r3c2:
    heatmap_data = pd.crosstab(fdf['Screen Time Period'], fdf['Attention Span'])
    heatmap_data = heatmap_data.reindex(period_order).fillna(0)[att_order]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Heatmap: Time Period vs Attention Span', fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
    ax.set_yticklabels([p.split(' (')[0] for p in period_order], rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Row 4: Subgroup Analysis ──────────────────────────────────
st.subheader("👥 Subgroup Analysis")
r4c1, r4c2 = st.columns(2)

with r4c1:
    occ_app = pd.crosstab(fdf['Occupation'], fdf['App_Category'])
    occ_app_pct = occ_app.div(occ_app.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    occ_app_pct.plot(kind='bar', ax=ax, edgecolor='white',
                     color=['#e74c3c','#9b59b6','#3498db','#e67e22','#2ecc71'])
    ax.set_title('App Usage: Students vs Professionals (%)', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='App', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with r4c2:
    age_order = ['Below 18', '18–24', '25–34', '35–44', '45 and above']
    age_att = fdf.groupby('Age Group')['Att_Score'].mean().reindex(age_order).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(age_att.index, age_att.values,
                  color=['#e74c3c','#e67e22','#3498db','#9b59b6','#2ecc71'], edgecolor='white')
    ax.axhline(fdf['Att_Score'].mean(), color='gray', linestyle='--', label='Avg')
    ax.set_title('Attention Score by Age Group', fontweight='bold')
    ax.set_ylabel('Avg Attention Score (1–4)')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    for bar, val in zip(bars, age_att.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Row 5: Correlation Heatmap ────────────────────────────────
st.subheader("🔢 Correlation Matrix")

app_map     = {'Social Media': 1, 'Streaming': 2, 'Messaging': 3, 'Gaming': 4, 'Productivity': 5}
period_map  = {'Morning (6 AM–12 PM)': 1, 'Afternoon (12 PM–6 PM)': 2,
               'Evening (6 PM–10 PM)': 3, 'Late night (10 PM–6 AM)': 4}

fdf2 = fdf.copy()
fdf2['App_Num']     = fdf2['App_Category'].map(app_map)
fdf2['Period_Num']  = fdf2['Screen Time Period'].map(period_map)
fdf2['Gender_Num']  = fdf2['Gender'].map({'Female': 0, 'Male': 1})
fdf2['Occ_Num']     = fdf2['Occupation'].map({'Student': 0, 'Professional': 1})

corr_df = fdf2[['Screen_Hours','App_Num','Period_Num','Att_Score',
                 'Prod_Score','Gender_Num','Occ_Num']].copy()
corr_df.columns = ['Screen Hours','App Type','Time Period',
                   'Attention','Productivity','Gender','Occupation']

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 10})
ax.set_title('Correlation Matrix (Lower Triangle)', fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
EDA of Screen Time Patterns & Impact on Student Attention Spans | Batch F-13 |
Yash Singhal · Krishna Kumar Karnani · Shakti Sidhu · Himani Kumari
</div>
""", unsafe_allow_html=True)
