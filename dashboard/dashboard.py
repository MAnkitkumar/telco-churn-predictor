"""
Telco Churn Analytics Dashboard — Plotly Dash
Run: python dashboard/dashboard.py
Opens at: http://localhost:8050
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'churn_cleaned.csv')
df = pd.read_csv(DATA_PATH)

# Readable churn label
df['Churn'] = df['Churn Value'].map({1: 'Churned', 0: 'Retained'})

# ── Colour palette ────────────────────────────────────────────────────────────
CHURN_COLORS   = {'Churned': '#f85149', 'Retained': '#3fb950'}
BG             = '#0f1117'
CARD_BG        = '#1c2130'
BORDER         = '#2a2f3e'
TEXT           = '#e6edf3'
SUBTEXT        = '#8b949e'
ACCENT         = '#58a6ff'

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family='Inter, sans-serif'),
    margin=dict(t=40, b=40, l=40, r=20),
)

# ── KPI helpers ───────────────────────────────────────────────────────────────
total      = len(df)
churned    = df['Churn Value'].sum()
churn_rate = churned / total
avg_monthly_churned = df[df['Churn Value'] == 1]['Monthly Charges'].mean()

def kpi_card(label, value, color=ACCENT):
    return html.Div([
        html.Div(label, style={'fontSize': '11px', 'color': SUBTEXT,
                               'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Div(value, style={'fontSize': '28px', 'fontWeight': '800', 'color': color}),
    ], style={
        'background': CARD_BG, 'border': f'1px solid {BORDER}',
        'borderRadius': '12px', 'padding': '20px 24px',
        'textAlign': 'center', 'flex': '1',
    })

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, title='Churn Intelligence Dashboard')

app.layout = html.Div(style={'background': BG, 'minHeight': '100vh',
                              'fontFamily': 'Inter, sans-serif', 'padding': '32px'}, children=[

    # Header
    html.Div([
        html.H1('📡 Churn Intelligence Dashboard',
                style={'color': TEXT, 'fontSize': '28px', 'fontWeight': '800', 'margin': '0'}),
        html.P('IBM Telco Customer Churn — Exploratory & Business Analysis',
               style={'color': SUBTEXT, 'marginTop': '6px'}),
    ], style={'marginBottom': '28px'}),

    # Filter bar
    html.Div([
        html.Div([
            html.Label('Contract Type', style={'color': SUBTEXT, 'fontSize': '12px'}),
            dcc.Dropdown(
                id='contract-filter',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': c, 'value': c} for c in sorted(df['Contract'].unique())],
                value='All',
                clearable=False,
                style={'background': CARD_BG, 'color': TEXT, 'border': f'1px solid {BORDER}'},
            ),
        ], style={'width': '220px'}),
        html.Div([
            html.Label('Internet Service', style={'color': SUBTEXT, 'fontSize': '12px'}),
            dcc.Dropdown(
                id='internet-filter',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': i, 'value': i} for i in sorted(df['Internet Service'].unique())],
                value='All',
                clearable=False,
                style={'background': CARD_BG, 'color': TEXT, 'border': f'1px solid {BORDER}'},
            ),
        ], style={'width': '220px'}),
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '24px'}),

    # KPI strip
    html.Div(id='kpi-strip', style={'display': 'flex', 'gap': '16px', 'marginBottom': '28px'}),

    # Row 1: Donut + Churn by Contract
    html.Div([
        html.Div(dcc.Graph(id='donut-chart'), style={
            'flex': '1', 'background': CARD_BG, 'borderRadius': '12px',
            'border': f'1px solid {BORDER}', 'padding': '16px'}),
        html.Div(dcc.Graph(id='contract-bar'), style={
            'flex': '2', 'background': CARD_BG, 'borderRadius': '12px',
            'border': f'1px solid {BORDER}', 'padding': '16px'}),
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

    # Row 2: Tenure histogram + Monthly charges box
    html.Div([
        html.Div(dcc.Graph(id='tenure-hist'), style={
            'flex': '1', 'background': CARD_BG, 'borderRadius': '12px',
            'border': f'1px solid {BORDER}', 'padding': '16px'}),
        html.Div(dcc.Graph(id='monthly-box'), style={
            'flex': '1', 'background': CARD_BG, 'borderRadius': '12px',
            'border': f'1px solid {BORDER}', 'padding': '16px'}),
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

    # Row 3: Internet service bar + Payment method bar
    html.Div([
        html.Div(dcc.Graph(id='internet-bar'), style={
            'flex': '1', 'background': CARD_BG, 'borderRadius': '12px',
            'border': f'1px solid {BORDER}', 'padding': '16px'}),
        html.Div(dcc.Graph(id='payment-bar'), style={
            'flex': '1', 'background': CARD_BG, 'borderRadius': '12px',
            'border': f'1px solid {BORDER}', 'padding': '16px'}),
    ], style={'display': 'flex', 'gap': '20px'}),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output('kpi-strip',     'children'),
    Output('donut-chart',   'figure'),
    Output('contract-bar',  'figure'),
    Output('tenure-hist',   'figure'),
    Output('monthly-box',   'figure'),
    Output('internet-bar',  'figure'),
    Output('payment-bar',   'figure'),
    Input('contract-filter',  'value'),
    Input('internet-filter',  'value'),
)
def update_all(contract, internet):
    dff = df.copy()
    if contract != 'All':
        dff = dff[dff['Contract'] == contract]
    if internet != 'All':
        dff = dff[dff['Internet Service'] == internet]

    t = len(dff)
    c = dff['Churn Value'].sum()
    rate = c / t if t else 0
    avg_m = dff[dff['Churn Value'] == 1]['Monthly Charges'].mean()

    # KPIs
    kpis = html.Div([
        kpi_card('Total Customers',          f'{t:,}'),
        kpi_card('Churned',                  f'{c:,}',          '#f85149'),
        kpi_card('Churn Rate',               f'{rate:.1%}',     '#f85149'),
        kpi_card('Avg Monthly (Churned)',     f'${avg_m:.2f}',   '#e3b341'),
    ], style={'display': 'flex', 'gap': '16px'})

    # Donut
    counts = dff['Churn'].value_counts().reset_index()
    counts.columns = ['Churn', 'Count']
    donut = px.pie(counts, names='Churn', values='Count', hole=0.55,
                   color='Churn', color_discrete_map=CHURN_COLORS,
                   title='Churn Split')
    donut.update_layout(**LAYOUT_BASE)
    donut.update_traces(textinfo='percent+label')

    # Contract bar
    ct = dff.groupby('Contract')['Churn Value'].mean().reset_index()
    ct.columns = ['Contract', 'Churn Rate']
    ct['Churn Rate %'] = (ct['Churn Rate'] * 100).round(1)
    contract_fig = px.bar(ct, x='Contract', y='Churn Rate', text='Churn Rate %',
                          title='Churn Rate by Contract Type',
                          color='Churn Rate', color_continuous_scale=['#3fb950', '#e3b341', '#f85149'])
    contract_fig.update_traces(texttemplate='%{text}%', textposition='outside')
    contract_fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)

    # Tenure histogram
    tenure_fig = px.histogram(dff, x='Tenure Months', color='Churn',
                               color_discrete_map=CHURN_COLORS, barmode='overlay',
                               nbins=36, opacity=0.75,
                               title='Tenure Distribution by Churn',
                               labels={'Tenure Months': 'Tenure (months)'})
    tenure_fig.update_layout(**LAYOUT_BASE)

    # Monthly charges box
    box_fig = px.box(dff, x='Churn', y='Monthly Charges',
                     color='Churn', color_discrete_map=CHURN_COLORS,
                     title='Monthly Charges vs Churn')
    box_fig.update_layout(**LAYOUT_BASE, showlegend=False)

    # Internet service bar
    inet = dff.groupby('Internet Service')['Churn Value'].mean().reset_index()
    inet.columns = ['Internet Service', 'Churn Rate']
    inet_fig = px.bar(inet, x='Internet Service', y='Churn Rate',
                      title='Churn Rate by Internet Service',
                      color='Churn Rate', color_continuous_scale=['#3fb950', '#f85149'])
    inet_fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)

    # Payment method bar
    pay = dff.groupby('Payment Method')['Churn Value'].mean().reset_index()
    pay.columns = ['Payment Method', 'Churn Rate']
    pay_fig = px.bar(pay, x='Churn Rate', y='Payment Method', orientation='h',
                     title='Churn Rate by Payment Method',
                     color='Churn Rate', color_continuous_scale=['#3fb950', '#f85149'])
    pay_fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)

    return kpis, donut, contract_fig, tenure_fig, box_fig, inet_fig, pay_fig


if __name__ == '__main__':
    app.run(debug=True, port=8050)
