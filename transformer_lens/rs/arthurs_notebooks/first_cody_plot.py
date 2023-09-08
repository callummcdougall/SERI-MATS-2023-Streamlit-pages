#%%

import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch as t

per_head_logit_diff = torch.tensor([[ 1.4349e-03, -7.9926e-04, -2.3806e-03, -4.5300e-03, -8.1502e-04,
          5.3822e-03,  1.3644e-02, -8.0567e-04, -2.1051e-02,  6.7343e-03,
          4.7915e-03, -4.2134e-03],
        [ 1.3898e-03,  1.8279e-03,  3.3699e-04, -5.6380e-03,  1.0027e-02,
         -6.3613e-03, -1.4256e-03, -7.6912e-03,  2.5773e-03,  4.9995e-03,
         -3.5043e-03, -1.1969e-03],
        [-7.7459e-03, -3.0683e-03, -1.8924e-03,  9.6281e-03, -3.3109e-06,
          1.7830e-03,  2.8216e-03,  4.1607e-03,  1.2925e-02, -6.3020e-03,
          4.2392e-03,  2.7941e-04],
        [ 4.8675e-03,  2.3966e-03,  6.3327e-03,  4.0890e-03, -4.7205e-03,
         -1.1839e-03, -8.1792e-03, -2.9606e-03, -1.2388e-03,  1.2024e-03,
         -1.8207e-03,  6.9916e-03],
        [-1.5789e-03,  1.5040e-03, -3.6474e-03, -4.2013e-03,  4.5219e-05,
         -1.2061e-03,  3.4501e-04, -8.4504e-04, -1.8408e-03,  3.2419e-03,
         -1.1715e-03,  1.2868e-02],
        [-8.3071e-03,  3.0215e-03, -2.5198e-03,  2.0679e-03, -2.5126e-04,
         -4.2871e-04,  1.3866e-03,  1.5363e-03, -6.8684e-03, -9.8826e-05,
         -3.1796e-02,  8.7770e-03],
        [-9.8706e-03, -1.1192e-04,  3.1522e-03, -3.4553e-03, -1.1143e-02,
          4.6752e-03,  1.0515e-02, -9.4824e-03, -1.1176e-03,  3.6180e-03,
          2.0960e-03,  5.2441e-03],
        [ 5.4126e-03, -1.4543e-03,  1.7166e-03,  7.7816e-02, -6.4828e-04,
         -1.5885e-02, -1.7865e-02,  1.5767e-03, -3.7882e-03,  1.7183e-01,
         -1.6627e-03, -4.8082e-03],
        [ 6.9502e-03, -1.8512e-04, -4.5380e-02,  2.5013e-02,  4.5767e-03,
         -8.1227e-03, -1.2084e-01,  8.9312e-04, -4.2207e-02, -6.6006e-03,
          2.6729e-01, -9.3043e-03],
        [ 1.1616e-01, -6.3648e-03,  9.6487e-02,  1.3665e-02, -1.1147e-02,
          2.3200e-02,  1.5661e+00,  7.1862e-02,  2.9995e-02,  3.1375e+00,
         -5.6879e-03, -1.0250e-02],
        [ 8.4218e-01,  2.1191e-01,  9.8512e-02, -7.5705e-03, -1.3834e-03,
          2.8189e-03,  2.8342e-01, -2.0460e+00, -1.9850e-03,  5.3798e-03,
          6.1978e-01,  1.4110e-02],
        [ 1.8143e-02, -8.7399e-02, -4.3477e-01,  3.8918e-02,  1.0356e-02,
         -2.6068e-03, -8.9001e-02,  3.2037e-02,  2.5302e-02,  1.4131e-02,
         -1.0784e+00,  3.9317e-02]], device='cuda:0')

path_patch_lds = torch.tensor([[ 1.4349e-03, -7.9926e-04, -2.3806e-03, -4.5300e-03, -8.1502e-04,
          5.3822e-03,  1.3644e-02, -8.0567e-04, -2.1051e-02,  6.7343e-03,
          4.7915e-03, -4.2134e-03],
        [ 1.3898e-03,  1.8279e-03,  3.3699e-04, -5.6380e-03,  1.0027e-02,
         -6.3613e-03, -1.4256e-03, -7.6912e-03,  2.5773e-03,  4.9995e-03,
         -3.5043e-03, -1.1969e-03],
        [-7.7459e-03, -3.0683e-03, -1.8924e-03,  9.6281e-03, -3.3109e-06,
          1.7830e-03,  2.8216e-03,  4.1607e-03,  1.2925e-02, -6.3020e-03,
          4.2392e-03,  2.7941e-04],
        [ 4.8675e-03,  2.3966e-03,  6.3327e-03,  4.0890e-03, -4.7205e-03,
         -1.1839e-03, -8.1792e-03, -2.9606e-03, -1.2388e-03,  1.2024e-03,
         -1.8207e-03,  6.9916e-03],
        [-1.5789e-03,  1.5040e-03, -3.6474e-03, -4.2013e-03,  4.5219e-05,
         -1.2061e-03,  3.4501e-04, -8.4504e-04, -1.8408e-03,  3.2419e-03,
         -1.1715e-03,  1.2868e-02],
        [-8.3071e-03,  3.0215e-03, -2.5198e-03,  2.0679e-03, -2.5126e-04,
         -4.2871e-04,  1.3866e-03,  1.5363e-03, -6.8684e-03, -9.8826e-05,
         -3.1796e-02,  8.7770e-03],
        [-9.8706e-03, -1.1192e-04,  3.1522e-03, -3.4553e-03, -1.1143e-02,
          4.6752e-03,  1.0515e-02, -9.4824e-03, -1.1176e-03,  3.6180e-03,
          2.0960e-03,  5.2441e-03],
        [ 5.4126e-03, -1.4543e-03,  1.7166e-03,  7.7816e-02, -6.4828e-04,
         -1.5885e-02, -1.7865e-02,  1.5767e-03, -3.7882e-03,  1.7183e-01,
         -1.6627e-03, -4.8082e-03],
        [ 6.9502e-03, -1.8512e-04, -4.5380e-02,  2.5013e-02,  4.5767e-03,
         -8.1227e-03, -1.2084e-01,  8.9312e-04, -4.2207e-02, -6.6006e-03,
          2.6729e-01, -9.3043e-03],
        [ 1.1616e-01, -6.3648e-03,  9.6487e-02,  1.3665e-02, -1.1147e-02,
          2.3200e-02,  1.5661e+00,  7.1862e-02,  2.9995e-02,  3.1375e+00,
         -5.6879e-03, -1.0250e-02],
        [ 1.9506e+00,  4.6548e-01,  6.2082e-01,  3.4350e-02, -2.6015e-03,
          1.9178e-03,  5.9527e-01,  2.0545e-01, -2.1982e-03, -1.5992e-03,
          1.4025e+00,  2.6782e-02],
        [ 1.6157e-02, -5.3783e-02,  7.6295e-01, -1.5097e-02,  6.7158e-03,
          5.6520e-03,  1.6998e-01,  2.8174e-02, -5.7994e-03,  4.3007e-02,
          4.3401e-01,  4.8612e-03]])

per_mlp_ld = torch.tensor([0.1623, 0.1304]).cuda()
path_patch_mlp_ld = torch.tensor([0.1623, 0.1310])

x = t.cat((per_head_logit_diff.flatten()[-24:], per_mlp_ld.flatten()), dim=0)
y = t.cat((path_patch_lds.flatten()[-24:], path_patch_mlp_ld.flatten()), dim=0)

pos_heads = ["L11H2", "L10H2", "L10H6", "L10H10", "L10H0"]
neg_heads = ["L10H7", "L11H10"]
text = [f"L{L}H{H}" for L in range(10, 12) for H in range(12)] + ["MLP10", "MLP11"]
colors = ["red" if t in neg_heads else "dodgerblue" if t in pos_heads else "black" if t.startswith("MLP") else "grey" for t in text]
pos_heads = ["L11H2", "", "", "L10H10", "L10H0"]
text = [t if t in neg_heads + pos_heads + ["L9H9", "L9H6"] else "" for t in text]
fig = px.scatter(width=1000, height=550)
# Add legend for Positive, Negative and other head and MLP

heads = {}
heads["dodgerblue"] = "Backup Head"
heads["red"] = "Negative Head"
heads["black"] = "MLP"
heads["grey"] = "Other Heads"

for color in heads.keys():
    filtered_x = [x for x, c in zip(x.cpu(), colors) if c == color]        
    filtered_y = [y for y, c in zip(y.cpu(), colors) if c == color]
    filtered_text = [t for t, c in zip(text, colors) if c == color]

    fig.add_trace(go.Scatter(
        x = filtered_x,
        y = filtered_y,
        text = filtered_text,
        textposition="top center",
        mode = 'markers+text',
        name =  heads[color],
        marker = dict(
            color = color,
            size = 12 if "blue" in color or "red" in color else 6,
        ),
    )
)

fig.add_trace(go.Scatter(
    x = [-0.01, 0.39],
    y = [0.82, 0.8],
    text = ["L10H2", "L10H6"],
    mode = 'text',
    showlegend=False,
))

x_range = np.linspace(start=-2.5, stop=2.5, num=100)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", line_dash='dash'))
fig.update_layout(
    paper_bgcolor='rgba(255,255,255,255)',
    plot_bgcolor='rgba(255,255,255,255)',
    # showlegend,
    title="Logit Difference Self-Repair in IOI",
    # title_font=dict(size=18),
    xaxis_title="Clean Logit Difference",
    yaxis_title="Post-Intervention Logit Difference",
    xaxis=dict(
        showline=True, linewidth=2, linecolor='black',
        showgrid=True, gridwidth=1, gridcolor='gray',
        # title_font=dict(size=16),
        range=[-2.5, 2.5]
    ),
    yaxis=dict(
        showline=True, linewidth=2, linecolor='black',
        showgrid=True, gridwidth=1, gridcolor='gray',
        # title_font=dict(size=16),
        range=[-2.5, 2.5]
    ),
    shapes=[
        dict(type='line', x0=0, x1=1, xref='paper', y0=0, y1=0, yref='y', line=dict(color='black', width=2)),
        dict(type='line', x0=0, x1=0, xref='x', y0=0, y1=1, yref='paper', line=dict(color='black', width=2))
    ],
    # legend=dict(title="Project:"),
    # font=dict(size=14),
    margin_r=50
)
fig.update_xaxes(range=[-2.5, 2.5])
fig.update_yaxes(range=[-2.5, 2.5])
fig.update_layout(font=dict(size=18),     legend=dict(
        title="Components:",
        x=1,  # x position
        y=0,  # y position
        xanchor='right',  # anchor point
        yanchor='bottom',  # anchor point
        bordercolor="Black",
        borderwidth=2
    ),)
fig.update_xaxes(showgrid=True, linecolor='rgba(128, 128, 128, 0.5)', linewidth=1)
fig.update_yaxes(showgrid=True, linecolor='rgba(128, 128, 128, 0.5)', linewidth=1)
fig.show()

# %%

fig.write_image("visualizing_self_repair.pdf")
# %%
