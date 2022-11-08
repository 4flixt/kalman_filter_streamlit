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

def main():
    tab1, tab2 = st.tabs(["One step scalar KF", "CSTR KF"])

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