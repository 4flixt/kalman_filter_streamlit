import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import expm
import pdb


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kf_streamlit
import prob_streamlit

def main():
    tab1, tab2 = st.tabs(["One step scalar KF", "CSTR KF"])

    with tab1:
        prob_streamlit.main()

    with tab2:
        kf_streamlit.main()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Kalman Filter",
        layout="wide"
    )

    main()