import streamlit as st

from streamlit_metrics import metric, metric_row
from streamlit_ace import st_ace

import pandas as pd
import numpy as np
import altair as alt
import nltk
nltk.download('stopwords')


st.set_page_config(
    page_title="Streamlit Sandbox",
    page_icon=":memo:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.sidebar.title(":memo: Editor settings")

st.title("Streamlit sandbox")
st.write("Prueba tus códigos de Streamlit desde el navegador sin instalar nada")

THEMES = [
    "ambiance",
    "chaos",
    "chrome",
    "clouds",
    "clouds_midnight",
    "cobalt",
    "crimson_editor",
    "dawn",
    "dracula",
    "dreamweaver",
    "eclipse",
    "github",
    "gob",
    "gruvbox",
    "idle_fingers",
    "iplastic",
    "katzenmilch",
    "kr_theme",
    "kuroir",
    "merbivore",
    "merbivore_soft",
    "mono_industrial",
    "monokai",
    "nord_dark",
    "pastel_on_dark",
    "solarized_dark",
    "solarized_light",
    "sqlserver",
    "terminal",
    "textmate",
    "tomorrow",
    "tomorrow_night",
    "tomorrow_night_blue",
    "tomorrow_night_bright",
    "tomorrow_night_eighties",
    "twilight",
    "vibrant_ink",
    "xcode",
]

KEYBINDINGS = ["emacs", "sublime", "vim", "vscode"]

editor, app = st.tabs(["Editor", "App"])

INITIAL_CODE = """
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

x = st.slider('Selecciona cantidad de datos a crear', min_value=2, max_value=40)
st.write('Datos a crear: ', x)

chart_data = pd.DataFrame(
    np.random.randn(x+1, 3),
    columns=['a', 'b', 'c'])

c = alt.Chart(chart_data).mark_circle().encode(
    x='a',
    y='b',
    size='c',
    color='c',
    tooltip=['a', 'b', 'c']
)

st.altair_chart(c, use_container_width=True)

st.write("Cargando dataset")
mc_df = pd.read_csv('McDonald_s_Reviews.csv', encoding="latin-1")
st.write(mc_df.head(10))
"""

with editor:
    code = st_ace(
        value=INITIAL_CODE,
        language="python",
        placeholder="st.header('Hello world!')",
        theme=st.sidebar.selectbox("Theme", options=THEMES, index=26),
        keybinding=st.sidebar.selectbox(
            "Keybinding mode", options=KEYBINDINGS, index=3
        ),
        font_size=st.sidebar.slider("Font size", 5, 24, 14),
        tab_size=4,
        wrap=st.sidebar.checkbox("Wrap lines", value=False),
        show_gutter=True,
        show_print_margin=True,
        auto_update=False,
        readonly=False,
        key="ace-editor",
    )
    st.write("Preciona `CTRL+ENTER` para refrescar")
    st.write("*Recuerda guardar tu code en otra parte!*")

with app:
    exec(code)

with st.sidebar:
    libraries_available = st.expander("Librerías instaladas")
    with libraries_available:
        st.write(
            """
        * Pandas
        * Altair
        * nltk
        """
        )
    dataset_available = st.expander("Datasets disponibles")
    with dataset_available:
        st.write(
            """
        * McDonald_s_Reviews.csv 
        * Bicycle_Chicago.csv
        * Airbnb_Locations.csv
        """
        )
