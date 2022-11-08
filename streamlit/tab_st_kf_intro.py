import streamlit as st
import pandas as pd
import numpy as np


def main():
    col_left, col_center, col_right, col_dummy = st.columns((2, 3, 1, 2))
    col_left.markdown("System dynamics:")
    col_left.latex(r"""
        \begin{aligned}
        x_{k+1} &= Ax_k + Bu_k + w\\
        y_k &= Cx_k + v
        \end{aligned}        
        """)
    col_left.markdown("With random variables:")
    col_left.latex(r"""
        \begin{aligned}
        x_0 &\sim \mathcal{N}(x_0; \hat x_0^{-}, P)\\
        w_k &\sim \mathcal{N}(w_k; 0, Q)\\
        v_k &\sim \mathcal{N}(v_k; 0, R)
        \end{aligned}        
        """)
    
    col_center.markdown("Kalman filter algorithm:")

    with col_center.expander("Correction step"):
        st.latex(r"""
            \begin{aligned}
                L_k &= P^-_k C^\top(CP^-_kC^\top + R)^{-1}\\
                \hat x_k &= \hat x_k^- + L_k(y_k-C\hat x_k^-)\\
                P_k &=(I-L_kC)P^-_k
            \end{aligned}        
            """)
    with col_center.expander("Prediction step"):
        st.latex(r"""
            \begin{aligned}
                \hat x_{k+1}^- &= A\hat x_k + Bu_k\\
                P^-_{k+1} &= A P_k A^\top + Q
            \end{aligned}        
            """)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Kalman filter",
        layout="wide"
    )

    main()