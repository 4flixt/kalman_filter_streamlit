# %% 

import streamlit as st
import numpy as np
from scipy.linalg import expm
import pdb


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kalmanfilter import KalmanFilter, get_sys

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']


def main():
    step_size = 0.1

    A, B, C, nx, nu, ny = get_sys(step_size)    

    KF = KalmanFilter(A, B, C, t_step=step_size, t_sim=30)

    x0 = np.array([.2, .2, .2]).reshape(-1,1)
    x0_observer = np.zeros((nx, 1))

    uk = np.array([4]).reshape(-1,1)


    col_left, col_center, col_right, col_dummy = st.columns((2, 3, 1, 2))

    n_options = 100
    options = np.logspace(-5, 0, n_options)
    col_right.markdown("Process noise $\sigma_q$")
    slide_sigma_q = col_right.select_slider(
            label="sigma_q (process noise)",
            label_visibility = 'collapsed',
            options = options,
            format_func = lambda x: f'{x:.2e}',
            value=options[n_options//2],
        )
    col_right.markdown("Meas noise $\sigma_r$")
    slide_sigma_r = col_right.select_slider(
            label="sigma_r (measurement noise)",
            options = options,
            label_visibility = 'collapsed',
            format_func = lambda x: f'{x:.2e}',
            value=options[n_options//2],
        )

    col_right.markdown("Initial state $\sigma_p$")
    slide_sigma_p = col_right.select_slider(
            label="sigma_p (initial state covariance)",
            options = options,
            label_visibility = 'collapsed',
            format_func = lambda x: f'{x:.2e}',
            value=options[n_options//2],
        )
    
    col_left.markdown("System dynamics:")
    col_left.latex(r"""
        \begin{aligned}
        \frac{d c_A}{dt} & =  \frac{\dot V }{V_R}(c_{A0}-c_A)-k_{AB}c_A\\
        \frac{d c_B}{dt} &= -\frac{\dot V }{V_R} c_B + k_{AB} c_A + k_{CB} c_C -
        k_{BC} c_B\\
        \frac{d c_C}{dt} &= -\frac{\dot V}{V_{R}} c_C + k_{BC} c_B - k_{CB} c_C
        \end{aligned}        
        """)
    col_left.markdown("Reformulated to:")
    col_left.latex(r"""
        \begin{aligned}
        x_{k+1} &= Ax_k + Bu_k + w_k\\
        y_k &= Cx_k + v_k
        \end{aligned}        
        """)
    col_left.markdown("With random variables:")
    col_left.latex(r"""
        \begin{aligned}
        x_0 &\sim \mathcal{N}(0, \sigma_p^2 I)\\
        w_k &\sim \mathcal{N}(0, \sigma_q^2 I)\\
        v_k &\sim \mathcal{N}(0, \sigma_r^2 I)
        \end{aligned}        
        """)


    #col_center.markdown("Estimated state vs. true state")

    KF.run(x0, x0_observer, slide_sigma_q, slide_sigma_r, slide_sigma_p, uk)

    # Create figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

    for i in range(nx):
        sig_x_i = np.sqrt(KF.P_data[:, i, i])
        p_3sig = (KF.x_data_observer[:, i, 0] + 3 * sig_x_i).flatten()
        m_3sig = (KF.x_data_observer[:, i, 0] - 3 * sig_x_i).flatten()

        fig.add_trace(
            go.Scatter(
                x=KF.time, y=m_3sig, name='c_{}'.format('abc'[i]),
                line = dict(
                    width=0, 
                    #color=DEFAULT_PLOTLY_COLORS[1]
                    ),
                showlegend=False,
                ),
            row=i+1, col=1)
        fig.add_trace(
            go.Scatter(
                x=KF.time, y=p_3sig, name='c_{}'.format('abc'[i]), fill='tonexty', 
                legendgroup='pm3sigma', legendgrouptitle_text='+/- 3 sigma'.format('abc'[i]),
                line = dict(
                    width=0, 
                    #color=DEFAULT_PLOTLY_COLORS[1]
                    ),
                ),
            row=i+1, col=1)

        fig.add_trace(
            go.Scatter(
                x=KF.time, y=KF.x_data[:, i, 0], 
                line = dict(
                    dash='dot',
                    width=2,
                    ),
                name='c_{}'.format('abc'[i]),
                legendgroup='true', legendgrouptitle_text='true states'
                ),
            row=i+1, col=1)
        fig.add_trace(
            go.Scatter(
                x=KF.time, y=KF.x_data_observer[:, i, 0],
                line = dict(
                    width=2,
                    ),
                #line_color=DEFAULT_PLOTLY_COLORS[1],
                name='c_{}'.format('abc'[i]),
                legendgroup='est', legendgrouptitle_text='est states'
                ),
            row=i+1, col=1)

    fig.add_trace(
        go.Scatter(
            x=KF.time, y=KF.y_data[:, 0, 0], name='measured',
            marker=dict(
                size=4, 
                #color=DEFAULT_PLOTLY_COLORS[4], 
                opacity=0.8),
            mode='markers',
            ),
        row=2, col=1)

    fig.update_xaxes(title_text="time [s]", row=3, col=1)
    fig.update_yaxes(title_text="c_a [kmol/l]", row=1, col=1)
    fig.update_yaxes(title_text="c_b [kmol/l]", row=2, col=1)
    fig.update_yaxes(title_text="c_c [kmol/l]", row=3, col=1)

    fig.update_layout(height=800, template='plotly_dark',
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    col_center.plotly_chart(fig, use_container_width=True)



if __name__ == '__main__':
    st.set_page_config(
        page_title="Kalman Filter",
        layout="wide"
    )

    main()