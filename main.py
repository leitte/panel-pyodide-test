# main.py
import io
import panel as pn
import pandas as pd
import numpy as np
from panel.template import FastListTemplate
import holoviews as hv

# Make sure Panel extensions loaded
pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh')

# widgets / content
slider = pn.widgets.FloatSlider(name="Amplitude", start=0, end=10, value=5)
text = pn.bind(lambda v: f"Amplitude is: {v:.2f}", slider)

file_input = pn.widgets.FileInput(accept=".csv", name="Upload a CSV")

data = {"group": np.random.randint(0, 10, 100), "value": np.random.randn(100)}
box = hv.Scatter(data, kdims="group", vdims="value").sort().opts()
hv_pane = pn.pane.HoloViews(box, height=300, sizing_mode="stretch_width")

main_content = pn.Column(
    pn.pane.Markdown("## My FastListTemplate App"),
    slider,
    pn.panel(text),
    file_input,
    hv_pane
)

def _on_file_change(event):
    #print(event)
    buffer = io.BytesIO(event.new)
    df = pd.read_csv(buffer)
    df['Time'] = pd.to_datetime(df.Time, format='%H:%M:%S')

    p = [hv.Curve(df, 'Time', tag, label=tag) for tag in ['T702', 'T704']]
    overlay = hv.Overlay(p)
    overlay.opts(xlabel='', ylabel='', 
                    show_grid=True, legend_position='top_left', 
                    #hooks=[hook]
                    )
    hv_pane.object = overlay

file_input.param.watch(_on_file_change, "value")

# # Build the template
# template = FastListTemplate(
#     title="Demo FastListTemplate",
#     sidebar=[pn.pane.Markdown("### Controls"), slider],
#     main=[main_content]
# )

# Explicitly write the template into the DOM element with id="app"
# In a Pyodide context we await this call so the frontend receives the view.
await pn.io.pyodide.write("app", main_content)
