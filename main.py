# main.py
import io
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv

# Make sure Panel extensions loaded
pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh')

hv.opts.defaults(hv.opts.Curve(color=hv.Cycle(["#145592","#0cad7f","#ffc43a","#EB5B67","#592835","#00b0be"])),
                 hv.opts.Violin(cmap=["#145592","#0cad7f","#ffc43a","#EB5B67","#592835","#00b0be"])
                 )

def hook(plot, element):
    # print('plot.state:   ', plot.state)
    # print('plot.handles: ', sorted(plot.handles.keys()))
    plot.handles['xaxis'].major_label_text_color = '#71717A'
    plot.handles['yaxis'].major_label_text_color = '#71717A'
    plot.handles['xaxis'].axis_label_text_color = '#71717A'
    plot.handles['yaxis'].axis_label_text_color = '#71717A'
    plot.handles['xaxis'].formatter.context = None

    if plot.handles['plot'].legend:
        plot.handles['plot'].legend.label_text_color = '#333'
        plot.handles['plot'].legend.label_text_font_size = '9pt'
        plot.handles['plot'].legend.border_line_color = None
        plot.handles['plot'].legend.orientation = "horizontal"

def minimal_style_hook(plot, element):
    plot.handles['plot'].outline_line_color = None
    plot.handles['yaxis'].minor_tick_line_color = None

# widgets / content
slider = pn.widgets.FloatSlider(name="Amplitude", start=0, end=10, value=5)
text = pn.bind(lambda v: f"Amplitude is: {v:.2f}", slider)

file_input = pn.widgets.FileInput(accept=".csv", name="Upload a CSV")

data = {"group": np.random.randint(0, 10, 100), "value": np.random.randn(100)}
box = hv.Scatter(data, kdims="group", vdims="value").sort().opts()
hv_pane = pn.pane.HoloViews(box, height=300, sizing_mode="stretch_width")

sidebar = pn.Column(
    file_input,
    slider,
    pn.panel(text),
)

main_content = pn.Tabs(
    ('Data', pn.Column(hv_pane))
)

def time_series_plot(df, features):
    p = [hv.Curve(df, 'Time', tag, label=tag) for tag in features]
    overlay = hv.Overlay(p)
    overlay.opts(hv.opts.Curve(height=250, responsive=True, 
                               active_tools=[]))
    overlay.opts(xlabel='', ylabel='', 
                    show_grid=True, legend_position='top_left', 
                    hooks=[hook, minimal_style_hook]
                    )
    return overlay

def _on_file_change(event):
    #print(event)
    buffer = io.BytesIO(event.new)
    df = pd.read_csv(buffer)
    df['Time'] = pd.to_datetime(df.Time, format='%H:%M:%S')

    temp_sensors = ['T703', 'T709', 'T711', 'T712', 'T705']
    temp_heaters = ['T701', 'T702', 'T704', 'T706', 'T708']
    pressure_difference = ['PDI701', 'PDI702']
    pressure_sensor = ['PY23']
    flow_transport = ['FT703', 'FT704']

    plots = [time_series_plot(df, f) for f in [temp_sensors, temp_heaters, pressure_difference, pressure_sensor, flow_transport]]
    layout = hv.Layout(plots).cols(1)
    layout.opts(sizing_mode="stretch_width")
    hv_pane.object = layout

file_input.param.watch(_on_file_change, "value")

# # Build the template
# template = FastListTemplate(
#     title="Demo FastListTemplate",
#     sidebar=[pn.pane.Markdown("### Controls"), slider],
#     main=[main_content]
# )

# Explicitly write the template into the DOM element with id="app"
# In a Pyodide context we await this call so the frontend receives the view.
await pn.io.pyodide.write("sidebar", sidebar)
await pn.io.pyodide.write("main", main_content)
