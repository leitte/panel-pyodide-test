# main.py
import io
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv

from js import Uint8Array, Float32Array, window as js_window
from pyodide.ffi import to_js
import asyncio

from scaler import RobustInputScaler

# Make sure Panel extensions loaded
pn.extension(sizing_mode="stretch_width")
hv.extension('bokeh')

hv.opts.defaults(hv.opts.Curve(color=hv.Cycle(["#145592","#0cad7f","#ffc43a","#EB5B67","#592835","#00b0be"])),
                 hv.opts.Violin(cmap=["#145592","#0cad7f","#ffc43a","#EB5B67","#592835","#00b0be"])
                 )

CURR_DF = None

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
model_input = pn.widgets.FileInput(accept=".onnx", name="Upload an onnx model")
model_status = pn.pane.Markdown("**Model:** not loaded")

data = {"group": np.random.randint(0, 10, 100), "value": np.random.randn(100)}
box = hv.Scatter(data, kdims="group", vdims="value").sort().opts()
hv_pane = pn.pane.HoloViews(box, height=300, sizing_mode="stretch_width")

sidebar = pn.Column(
    file_input,
    model_input,
    model_status,
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
    CURR_DF = df

    temp_sensors = ['T703', 'T709', 'T711', 'T712', 'T705']
    temp_heaters = ['T701', 'T702', 'T704', 'T706', 'T708']
    pressure_difference = ['PDI701', 'PDI702']
    pressure_sensor = ['PY23']
    flow_transport = ['FT703', 'FT704']

    plots = [time_series_plot(df, f) for f in [temp_sensors, temp_heaters, pressure_difference, pressure_sensor, flow_transport]]
    layout = hv.Layout(plots).cols(1)
    layout.opts(sizing_mode="stretch_width")
    hv_pane.object = layout


async def run_onnx_inference(arr_np: np.ndarray):
    """
    arr_np: numpy array of shape (1, channels, seq_len), dtype float32
    returns: (out_np, ms)
    NOTE: arr_np is not modified.
    """
    # Make sure we have float32, but don't touch arr_np itself
    arr = np.asarray(arr_np, dtype=np.float32)
    shape = list(arr.shape)             # e.g. [1, C, T]
    print(shape)

    # Flatten for transport (view, not modifying arr_np)
    flat = arr.reshape(-1)

    # Convert to JS types
    flat_js = to_js(flat)               # -> JS Float32Array
    shape_js = to_js(shape)             # -> JS Array<number>

    # Call the JS helper; it expects (typedArray, dimsArray)
    res = await js_window.ort_helpers.runModel(flat_js, shape_js)

    # res is a JsProxy with properties data, dims, ms
    # Use the documented .to_py() method on JsProxy
    out_data = res.data.to_py()         # Python list of floats
    out_dims = res.dims.to_py()         # Python list of ints
    ms = float(res.ms)

    out_np = np.array(out_data, dtype=np.float32).reshape(out_dims)
    return out_np, ms

async def apply_onnx_to_df(df: pd. DataFrame):
    model_status.object = "**Model:** running..."
    try:
        channels = 18
        seq_len  = 140
        arr = np.random.randn(1, seq_len, channels).astype(np.float32)

        features = ['LS701', 'LS702', 'T701', 'T702', 'T703', 'T704', 'T706', 'T708', 'T709', 'T711', 'T712', 'T705', 'FT703', 'FT704', 'PDI701', 'PDI702', 'PY23', 'FYI702']
        arr = df.loc[:seq_len,features].to_numpy().astype(np.float32).T
        arr = np.expand_dims(arr, axis=0)           # shape [1, window, features]

        out_np, ms = await run_onnx_inference(arr)
        print("ONNX output shape:", out_np.shape)
        print("ONNX output sample:", out_np.flatten()[:10])
    except Exception as e:
        model_status.object = f"**Model:** inference failed — {e}"
        raise

async def load_model_async(data):
        model_bytes = Uint8Array.new(data)
        try:
            res = await js_window.ort_helpers.createSession(model_bytes)
            model_status.object = f"**Model:** loaded (backend: {res.backend})"
        except Exception as e:
            model_status.object = f"**Model:** failed to load — {e}"

def _on_model_change(event):
    async def _do_work(event):
        await load_model_async(event.new)
        print('#################')
        print(CURR_DF)
        print("dataset", CURR_DF.shape if isinstance(CURR_DF, pd.DataFrame) else 'no dataset')
        if CURR_DF:
            await apply_onnx_to_df(CURR_DF)
        else:
            await apply_onnx_to_df(CURR_DF)

    asyncio.ensure_future(_do_work(event))


file_input.param.watch(_on_file_change, "value")
model_input.param.watch(_on_model_change, "value")

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
