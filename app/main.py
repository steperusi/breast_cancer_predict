import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv(r'model/reduced_data.csv', index_col=0)

    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data

def add_sidebar():
    st.sidebar.header('Cell Nuclei Mesurement')

    data = get_clean_data()

    slider_labels = [
        ("Smoothness (mean)", "smoothness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Smoothness (se)", "smoothness_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Area (worst)", "area_worst"),
        ("Concave points (worst)", "concave points_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    x = data.drop(columns=['diagnosis'])

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = [
        'Smoothness (mean)', 'Concavity (mean)', 'Concave points (mean)',
        'Fractal dimension (mean)', 'Smoothness (se)', 'Symmetry (se)',
        'Area (worst)', 'Concave points (worst)'
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['smoothness_mean'], input_data['concavity_mean'], input_data['concave points_mean'],
            input_data['fractal_dimension_mean'], input_data['smoothness_se'], input_data['symmetry_se'],
            input_data['area_worst'], input_data['concave points_worst']
        ],
        theta=categories,
        fill='toself',
        line_color='blue'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False
            )),
        showlegend=False
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open(r'model/model.pkl', 'rb'))
    scaler = pickle.load(open(r'model/scaler.pkl', 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader('Cell cluster prediction')
    st.write('The cell cluster is:')

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write('Probability of being benign: ', model.predict_proba(input_array_scaled)[0][0])
    st.write('Probability of being malicious: ', model.predict_proba(input_array_scaled)[0][1])

    st.write('This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.')

def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon=':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    with open(r'app/style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write('Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.')

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart =  get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()