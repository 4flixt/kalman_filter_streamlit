import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import expm
import pdb


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tab_st_kf_1d
import tab_st_kf_cstr
import tab_st_kf_intro

def main():
    tab0, tab1, tab2 = st.tabs(["Intro", "KF distributions", "CSTR example"])

    with tab0:
        tab_st_kf_intro.main()

    with tab1:
        tab_st_kf_1d.main()

    with tab2:
        tab_st_kf_cstr.main()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Kalman filter",
        layout="wide"
    )

    main()