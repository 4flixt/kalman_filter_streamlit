import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import expm
import pdb

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kalmanfilter

def main():

    col_left, col_center, col_right, col_dummy = st.columns((2, 3, 1, 2))


    col_left.markdown("System dynamics:")
    col_left.latex(r"""
        \begin{aligned}
        x_{1} &= 2x_0 + w_0\\
        y_0 &= x_0 + v_0
        \end{aligned}        
        """)
    col_left.markdown("With random variables:")
    col_left.latex(r"""
        \begin{aligned}
        x_0 &\sim \mathcal{N}(x_0; \hat x_0^{-}, \sigma_p^2)\\
        w_0 &\sim \mathcal{N}(w_0; 0, \sigma_q^2)\\
        v_0 &\sim \mathcal{N}(v_0; 0, \sigma_r^2)
        \end{aligned}        
        """)
    col_left.markdown("Important distributions:")
    col_left.latex(r"""
        \begin{aligned}
        p(x_0) &= \mathcal{N}(x_0; \hat x_0^{-1}, \sigma_p^2)\\
        p(y_0|x_0) &= \mathcal{N}(y_0; x_0, \sigma_r^2)\\
        p(x_0|y_0) &= \frac{p(y_0|x_0)p(y_0)}{p(y_0)}\\
        x_1 &= 2x_0 + w_0
        \end{aligned}        
        """)

    multiselect_options = col_right.multiselect(
        'Show',
        ['p(y0|x0)', 'p(x0)', 'p(x0|y0)', 'p(x1)'],
        ['p(x0)']
    )


    n_options = 100
    
    options_y = np.linspace(0, 3, n_options)
    col_right.markdown("Measurement $y$")
    slide_mu_y = col_right.select_slider(
            label="mu_meas",
            label_visibility = 'collapsed',
            options = options_y,
            format_func = lambda x: f'{x:.2f}',
            value=options_y[n_options//2],
        )

    options_sigma = np.logspace(-2, 0, n_options)

    col_right.markdown("Initial state $\sigma_p$")
    slide_sigma_p = col_right.select_slider(
            label="sigma_p (initial state covariance)",
            options = options_sigma,
            label_visibility = 'collapsed',
            format_func = lambda x: f'{x:.2e}',
            value=options_sigma[n_options//2],
        )

    col_right.markdown("Meas noise $\sigma_r$")
    slide_sigma_r = col_right.select_slider(
            label="sigma_r (measurement noise)",
            options = options_sigma,
            label_visibility = 'collapsed',
            format_func = lambda x: f'{x:.2e}',
            value=options_sigma[n_options//2],
        )

    col_right.markdown("Process noise $\sigma_q$")
    slide_sigma_q = col_right.select_slider(
            label="sigma_q (process noise)",
            label_visibility = 'collapsed',
            options = options_sigma,
            format_func = lambda x: f'{x:.2e}',
            value=options_sigma[n_options//2],
        )

    A = 2.0

    mu_x = 1
    Sigma_x = slide_sigma_p
    x0_dist = kalmanfilter.Normal(mu_x, Sigma_x)

    mu_y = slide_mu_y
    Sigma_y = slide_sigma_r
    y0_dist = kalmanfilter.Normal(mu_y, Sigma_y)

    w_dist = kalmanfilter.Normal(0, slide_sigma_q)
    
    x0_hat_dist = x0_dist*y0_dist
    x1_dist = x0_hat_dist*A + w_dist

    x_linspace = np.linspace(-2, 6, 200)
    x0_pdf = x0_dist.pdf(x_linspace)
    y0_pdf = y0_dist.pdf(x_linspace)
    x0_hat_pdf = x0_hat_dist.pdf(x_linspace)
    x1_pdf = x1_dist.pdf(x_linspace)


    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True)

    # Add traces
    if 'p(x0)' in multiselect_options:
        fig.add_trace(
            go.Scatter(x=x_linspace, y=x0_pdf, name='p(x0)', fill='tozeroy'),
            row=1, col=1)
    if 'p(y0|x0)' in multiselect_options:
        fig.add_trace(
            go.Scatter(x=x_linspace, y=y0_pdf, name='p(y0|x0)', fill='tozeroy'),
            row=1, col=1)
    if 'p(x0|y0)' in multiselect_options:
        fig.add_trace(
            go.Scatter(x=x_linspace, y=x0_hat_pdf, name='p(x0|y0)', fill='tozeroy'),
            row=1, col=1)

        fig.add_vline(x=x0_hat_dist.mu[0,0], 
            line_dash="dash", 
            line_width=2,
            line_color="white",
            row=1, col=1
            )

    if 'p(x1)' in multiselect_options:
        fig.add_trace(
            go.Scatter(x=x_linspace, y=x1_pdf, name='p(x1)', fill='tozeroy'),
            row=2, col=1)

        fig.add_vline(x=x0_hat_dist.mu[0,0], 
            line_dash="dash", 
            line_width=2,
            line_color="white",
            row=2, col=1
            )
        fig.add_vline(x=x1_dist.mu[0,0], 
            line_dash="solid", 
            name="x1",
            line_width=2,
            #line_color="white",
            row=2, col=1
            )

    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_yaxes(title_text="p( . )", row=1, col=1)
    fig.update_yaxes(title_text="p( . )", row=2, col=1)



    fig.update_layout(height=800, template='plotly_dark',
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    col_center.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    st.set_page_config(
        page_title="Kalman Filter",
        layout="wide"
    )

    main()


    
