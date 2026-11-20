from pathlib import Path
import base64
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.preprocessing import FEATURES, load_data
from src.predict import load_models, validate_frame, predict_frame
from src.history import add as history_add, read as history_read, clear as history_clear

ROOT=Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "glycolens-logo.svg"
LOGO_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
# Restore a user-selected appearance from the URL; new visits default to dark.
saved_theme = st.query_params.get("theme", "dark")
if saved_theme not in ("light", "dark"):
    saved_theme = "dark"
st.set_page_config(page_title="GlycoLens | Diabetes Analytics",page_icon=LOGO_DATA_URI,layout="wide",initial_sidebar_state="collapsed")
# The in-app segmented control is the single source of truth for page colors.
if "glycolens_theme" not in st.session_state:
    st.session_state.glycolens_theme = saved_theme

def remember_theme():
    """Persist the chosen appearance in the URL across reloads."""
    st.query_params["theme"] = st.session_state.glycolens_theme
theme_type = st.sidebar.segmented_control(
    "Appearance", ["light", "dark"], key="glycolens_theme",
    format_func=lambda value: "☀  Light" if value == "light" else "☾  Dark",
    default="dark", selection_mode="single", required=True, width="stretch",
    on_change=remember_theme,
)
THEMES = {
    "light": {"ink":"#183532","muted":"#5d716d","teal":"#087f72","line":"#dce7e4","paper":"#f5f8f7","surface":"#ffffff","surface-raised":"#edf5f2","sidebar":"#edf4f2","red":"#a43d46","green":"#16745f","notice":"#fff9ed","notice-border":"#e8dcc2","notice-ink":"#5f5031","result-positive-bg":"#fff1ef","result-negative-bg":"#edf7f2","plot":"#304b47","grid":"#dce7e4"},
    "dark": {"ink":"#e7f1ef","muted":"#a6bbb8","teal":"#66c7b5","line":"#304340","paper":"#101a19","surface":"#182321","surface-raised":"#1d2d2a","sidebar":"#14201e","red":"#f08d8b","green":"#6fc9a9","notice":"#322c1e","notice-border":"#5b4e2c","notice-ink":"#e2d3a7","result-positive-bg":"#382322","result-negative-bg":"#19332b","plot":"#d0dfdc","grid":"#354541"},
}
theme = THEMES.get(theme_type, THEMES["light"])
theme_vars = ";".join([f"color-scheme:{'dark' if theme_type == 'dark' else 'light'}"] + [f"--{key}:{value}" for key,value in theme.items()])
css = (ROOT/'assets/styles.css').read_text(encoding='utf-8')
st.markdown(f"<style>:root{{{theme_vars}}}{css}</style>",unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-brand topbar-brand"><img src="{LOGO_DATA_URI}" alt="GlycoLens logo"><div><div class="hero-brand-name">GlycoLens</div><div class="eyebrow">EDUCATIONAL HEALTH ANALYTICS</div></div></div>',
    unsafe_allow_html=True,
)
@st.cache_resource
def get_models(): return load_models()
@st.cache_data
def get_data(): return load_data(ROOT/'data/diabetes.csv')
@st.cache_data
def get_eval(): return json.loads((ROOT/'models/evaluation.json').read_text(encoding='utf-8'))
try:
    models=get_models(); metrics=get_eval(); df=get_data()
except Exception as e:
    st.error(f"Model assets are not ready: {e}. Run `python -m src.train` from the project directory.")
    st.stop()

page=st.sidebar.radio("Workspace",["Overview","Prediction","Model performance","Batch prediction","Prediction history","Interpretability"],label_visibility="collapsed",key="workspace_navigation")
st.sidebar.divider()
st.sidebar.caption("Pima Indians Diabetes dataset · 768 records")

scores=pd.DataFrame(metrics['models']).T
best_name=scores['f1'].astype(float).idxmax()
series = ["#66c7b5", "#73a9d1", "#e49a70", "#b29be5"] if theme_type == "dark" else ["#087f72", "#3b78a0", "#a15c35", "#7455a5"]
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)
def metric_card(label,value): st.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
def show_table(data, **kwargs):
    """Use Streamlit's theme-aware native dataframe instead of a fixed Styler."""
    st.dataframe(data, use_container_width=True, **kwargs)

def chart_theme(fig):
    # Streamlit exposes the active appearance to this session; keep Plotly's
    # SVG/canvas text in step with the app theme (CSS alone cannot recolor it).
    ink = theme["plot"]
    grid = theme["grid"]
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='DM Sans',color=ink),margin=dict(l=10,r=15,t=35,b=10),legend_title_text='', hoverlabel=dict(font=dict(color=ink)),legend=dict(font=dict(color=ink)))
    fig.update_xaxes(showgrid=False,zeroline=False,color=ink,linecolor=grid,tickfont=dict(color=ink),title_font=dict(color=ink))
    fig.update_yaxes(gridcolor=grid,zeroline=False,color=ink,linecolor=grid,tickfont=dict(color=ink),title_font=dict(color=ink))
    return fig

if page=="Overview":
    hero("A clearer view of model behavior.","Explore a reproducible diabetes-classification workflow—from dataset quality and evaluation to an educational, model-driven estimate.")
    c=st.columns(4)
    for col,(label,value) in zip(c,[("Dataset records",f"{len(df):,}"),("Measured features","08"),("Algorithms","03"),("Best test F1",f"{scores.loc[best_name,'f1']:.3f}")]):
        with col: metric_card(label,value)
    st.markdown('<div class="section-title">Performance at a glance</div>',unsafe_allow_html=True)
    left,right=st.columns([1.25,1])
    with left:
        long=scores.reset_index(names='Model').melt(id_vars='Model',value_vars=['precision','recall','f1','roc_auc'],var_name='Metric',value_name='Score')
        fig=px.bar(long,x='Model',y='Score',color='Metric',barmode='group',color_discrete_sequence=series,hover_data={'Score':':.3f'})
        chart_theme(fig).update_yaxes(range=[0,1],title='Held-out score'); st.plotly_chart(fig,use_container_width=True)
    with right:
        counts=df.Outcome.value_counts().sort_index()
        fig=px.bar(x=['Negative','Positive'],y=[counts.get(0,0),counts.get(1,0)],labels={'x':'Observed class','y':'Records'},color_discrete_sequence=[series[0]])
        chart_theme(fig).update_layout(title='Observed class balance'); st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"**Current reference model:** {best_name}, selected by held-out F1 among three models. F1 balances precision and recall; it does not encode the relative clinical cost of false positives and false negatives. Evaluation uses one stratified 20% test split (seed 42); metrics are illustrative and do not establish clinical performance.")
    show_table(scores[['accuracy','precision','recall','f1','roc_auc']].rename(columns=str.title).round(3))
elif page=="Prediction":
    hero("Individual prediction","Enter the eight measurements. The saved pipelines handle invalid-measurement imputation and scaling consistently with training.")
    defaults=[1,120,70,20,79,25.0,0.45,33]
    with st.form('prediction_form'):
        st.markdown("#### Patient measurements")
        cols=st.columns(4)
        vals={}
        specs=[("Pregnancies",0,30,1), ("Glucose",0,300,1), ("BloodPressure",0,200,1), ("SkinThickness",0,100,1), ("Insulin",0,1000,1), ("BMI",0.0,90.0,.1), ("DiabetesPedigreeFunction",0.0,3.0,.001), ("Age",1,120,1)]
        for i,(name,lo,hi,step) in enumerate(specs):
            with cols[i%4]:
                number_format = "%.3f" if name == "DiabetesPedigreeFunction" else None
                vals[name]=st.number_input(name,min_value=lo,max_value=hi,value=defaults[i],step=step,format=number_format,help="A zero measurement is treated as missing and imputed.")
        selected=st.selectbox("Primary model",list(models),index=list(models).index(best_name))
        compare=st.checkbox("Compare all three models")
        submitted=st.form_submit_button("Generate estimate",type="primary",use_container_width=True)
    if submitted:
        try:
            x=validate_frame(pd.DataFrame([vals]))
            names=list(models) if compare else [selected]
            answers=[]
            for name in names:
                model=models[name]; prediction=int(model.predict(x)[0]); probability=float(model.predict_proba(x)[0,1]); answers.append((name,prediction,probability))
                history_add(name,prediction,probability,vals)
            st.session_state['last_answers']=answers
        except Exception as e: st.error(str(e))
    if st.session_state.get('last_answers'):
        st.markdown("#### Model output")
        for name,pred,prob in st.session_state.last_answers:
            css='result-positive' if pred else 'result-negative'; label='Positive class (model predicts Outcome 1)' if pred else 'Negative class (model predicts Outcome 0)'
            st.markdown(f'<div class="{css}"><b>{label}</b><br>{name} · model probability for class 1: <b>{prob:.1%}</b></div>',unsafe_allow_html=True)
        st.caption("Probabilities are model outputs, not calibrated estimates of medical risk. Zero values in selected measurements are interpreted as missing and imputed.")
elif page=="Model performance":
    hero("Performance, with context.","Compare actual held-out test results. Threshold-dependent metrics describe different trade-offs; no score by itself validates a diagnostic tool.")
    show_table(scores[['accuracy','precision','recall','f1','roc_auc']].rename(columns=str.title).round(3))
    st.markdown("**Reading the trade-off:** recall measures the share of positive test cases found; higher recall can mean fewer false negatives but may increase false positives. Precision measures how often positive predictions were correct in this test set. Threshold choice is not clinically optimized.")
    selected_metric=st.selectbox("Metric to compare",['f1','roc_auc','recall','precision','accuracy'])
    perf=scores[selected_metric].astype(float).rename_axis('Model').reset_index(name='Score').sort_values('Score',ascending=False)
    fig=px.bar(perf,x='Model',y='Score',color_discrete_sequence=[series[0]],text=perf.Score.map(lambda x:f'{x:.3f}'),hover_data={'Score':':.3f'})
    chart_theme(fig).update_yaxes(range=[0,1]); st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Confusion matrices · held-out test split")
    cols=st.columns(3)
    for col,(name,row) in zip(cols,metrics['models'].items()):
        with col:
            st.markdown(f"**{name}**")
            cm=row['confusion_matrix']; fig=px.imshow(cm,text_auto=True,x=['Predicted 0','Predicted 1'],y=['Actual 0','Actual 1'],color_continuous_scale=[theme["surface"],series[0]],aspect='auto')
            chart_theme(fig).update_layout(coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### ROC curves")
    fig=go.Figure()
    for name,row in metrics['models'].items(): fig.add_trace(go.Scatter(x=row['roc']['fpr'],y=row['roc']['tpr'],mode='lines',name=name,hovertemplate='FPR %{x:.2f}<br>TPR %{y:.2f}<extra>'+name+'</extra>'))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Chance reference',line=dict(dash='dash',color='#97a8a8')))
    fig.update_layout(xaxis_title='False positive rate',yaxis_title='True positive rate'); st.plotly_chart(chart_theme(fig),use_container_width=True)
elif page=="Batch prediction":
    hero("Batch predictions","Upload a CSV with the eight feature columns. Model predictions are generated from the persisted pipelines, then exported with class probabilities.")
    st.download_button("Download example input CSV",pd.DataFrame([[1,120,70,20,79,25,.45,33]],columns=FEATURES).to_csv(index=False),'example_input.csv','text/csv')
    uploaded=st.file_uploader("Upload CSV",type=['csv'])
    if uploaded:
        try:
            incoming=pd.read_csv(uploaded)
            if len(incoming)==0: raise ValueError('The uploaded CSV contains no rows.')
            if len(incoming)>10000: raise ValueError('Please upload no more than 10,000 rows at a time.')
            predicted=predict_frame(incoming,models)
            predicted.insert(0,'Record',range(1,len(predicted)+1))
            st.success(f"Predictions complete for {len(predicted):,} records.")
            show_table(predicted,height=440)
            st.download_button("Download predictions CSV",predicted.to_csv(index=False),'diabetes_predictions.csv','text/csv')
        except Exception as e: st.error(str(e))
    st.caption("Outcome labels in an uploaded file are ignored; only required feature columns are used. Blank/non-numeric features are rejected. Zero in designated medical measurement fields is imputed.")
elif page=="Prediction history":
    hero("Prediction history","A local SQLite log of estimates made in this application. History is stored on this machine and is not a medical record.")
    entries=history_read()
    if entries:
        rows=[]
        for r in entries:
            rows.append({"Timestamp (UTC)":r['timestamp'],"Model":r['model'],"Class":'Positive' if r['outcome'] else 'Negative',"P(class 1)":r['probability'],**r['features']})
        show_table(pd.DataFrame(rows))
        confirm=st.checkbox("I understand this permanently clears the local prediction history")
        if st.button("Clear history",disabled=not confirm): history_clear(); st.rerun()
    else: st.info("No predictions recorded yet. Run an individual estimate to start a history.")
elif page=="Interpretability":
    hero("What influences the forest?","The chart shows Random Forest impurity-based feature importance. It is model-specific, can favor certain feature types, and is not a causal or medical explanation.")
    fi=pd.Series(metrics['feature_importance']).sort_values(ascending=True).rename_axis('Feature').reset_index(name='Importance')
    fig=px.bar(fi,x='Importance',y='Feature',orientation='h',color_discrete_sequence=[series[0]],hover_data={'Importance':':.3f'})
    chart_theme(fig).update_layout(yaxis_title='',xaxis_title='Impurity-based importance'); st.plotly_chart(fig,use_container_width=True)
    show_table(fi.sort_values('Importance',ascending=False).round({'Importance':4}))
    st.caption("Correlated features can share or distort importance; these values do not estimate effect direction, individual contribution, or causation.")

st.divider()
st.caption("GlycoLens · University ML demonstration · Pima Indians Diabetes dataset")
