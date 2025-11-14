# main.py
import io
import zipfile
import param
import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv

from js import Uint8Array, Float32Array, window as js_window
from pyodide.ffi import to_js
import asyncio

import sys
sys.path.insert(0, "/src")
from mycode.scaler import RobustInputScaler, RobustTargetScaler

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
class GUI:
    file_input: pn.widgets.FileInput
    model_input: pn.widgets.FileInput
    model_status: pn.pane.Markdown
    start_ad_button: pn.widgets.Button

    hv_pane: pn.pane.HoloViews

    sidebar: pn.Column
    main: pn.Tabs

    def __init__(self) -> None:
        self.slider = pn.widgets.FloatSlider(name="Amplitude", start=0, end=10, value=5)
        self.text = pn.bind(lambda v: f"Amplitude is: {v:.2f}", self.slider)

        self.file_input = pn.widgets.FileInput(accept=".csv", name="Upload a CSV")
        self.model_input = pn.widgets.FileInput(accept=".zip", name="Upload a model.zip")
        self.model_status = pn.pane.Markdown("**Model:** not loaded")
        self.start_ad_button = pn.widgets.Button(name="Detect Anomalies")

        data = {"group": np.random.randint(0, 10, 100), "value": np.random.randn(100)}
        box = hv.Scatter(data, kdims="group", vdims="value").sort().opts()
        self.hv_pane = pn.pane.HoloViews(box, height=300, sizing_mode="stretch_width")

        self._init_layout()

    def _init_layout(self):
        self.sidebar = pn.Column(
            self.file_input,
            self.model_input,
            self.model_status,
            self.start_ad_button,
            self.slider,
            pn.panel(self.text),
        )

        self.main = pn.Tabs(
            ('Data', pn.Column(self.hv_pane))
        )

    def _on_update_plots(self, layout: hv.Layout):
        self.hv_pane.object = layout


class DataHandler(param.Parameterized):
    message = param.String(default="")
    df = param.DataFrame(pd.DataFrame())

    input_scaler: RobustInputScaler
    target_scaler: RobustTargetScaler

    def __init__(self, **params):
        super().__init__(**params)

    def _on_file_change(self, event):
        buffer = io.BytesIO(event.new)
        df_new = pd.read_csv(buffer)
        df_new['Time'] = pd.to_datetime(df_new.Time, format='%H:%M:%S')
        self.df = df_new

        print('loaded dataset', self.df.shape)

    async def load_model_async(self, data):
        model_bytes = Uint8Array.new(data)
        try:
            res = await js_window.ort_helpers.createSession(model_bytes)
            self.message = f"loaded (backend: {res.backend})"
        except Exception as e:
            pass
            self.message = f"failed to load — {e}"    

    def _on_model_change(self, event):
        try:
            with zipfile.ZipFile(io.BytesIO(event.new)) as zf:
                print(zf.namelist())
                onnx_bytes = zf.read('model.onnx')
                input_scaler_bytes = zf.read('input_scaler.json')
                target_scaler_bytes = zf.read('target_scaler.json')

                # load model
                #asyncio.ensure_future(self.load_model_async(onnx_bytes))

                # setup scalers
                #import json
                #print(json.loads(input_scaler_bytes.decode("utf-8")))
                scaler = RobustInputScaler()
                scaler.loads(input_scaler_bytes.decode("utf-8"))
                self.input_scaler = scaler

        except Exception as e:
            print("zip not working", e)

class DataRenderer(param.Parameterized):
    features: list[str]
    time_series_plot = param.ClassSelector(class_=hv.Layout)

    def __init__(self, **params):
        super().__init__(**params)


    def create_time_series_plot(self, df, features):
        p = [hv.Curve(df, 'Time', tag, label=tag) for tag in features]
        overlay = hv.Overlay(p)
        overlay.opts(hv.opts.Curve(height=250, responsive=True, 
                                active_tools=[]))
        overlay.opts(xlabel='', ylabel='', 
                        show_grid=True, legend_position='top_left', 
                        hooks=[hook, minimal_style_hook]
                        )
        return overlay
    
    def update_plots(self, df):
        print("update plots")
        temp_sensors = ['T703', 'T709', 'T711', 'T712', 'T705']
        temp_heaters = ['T701', 'T702', 'T704', 'T706', 'T708']
        pressure_difference = ['PDI701', 'PDI702']
        pressure_sensor = ['PY23']
        flow_transport = ['FT703', 'FT704']

        plots = [self.create_time_series_plot(df, f) for f in [temp_sensors, temp_heaters, pressure_difference, pressure_sensor, flow_transport]]
        layout = hv.Layout(plots).cols(1)
        layout.opts(sizing_mode="stretch_width")
        self.time_series_plot = layout


class AnomalyDetector(param.Parameterized):
    message = param.String(default="**Model**")

    def __init__(self, **params):
        super().__init__(**params)

    async def load_model_async(self, data):
        model_bytes = Uint8Array.new(data)
        try:
            res = await js_window.ort_helpers.createSession(model_bytes)
            self.message = f"loaded (backend: {res.backend})"
        except Exception as e:
            pass
            self.message = f"failed to load — {e}"

    def update_model(self, model):
        async def _do_work(model):
            await self.load_model_async(model)

        asyncio.ensure_future(_do_work(model))

    async def apply_onnx_to_df(self, df: pd. DataFrame):
        self.message = "running..."
        try:
            channels = 18
            seq_len  = 140
            arr = np.random.randn(1, seq_len, channels).astype(np.float32)

            scaler = RobustInputScaler()

            features = ['LS701', 'LS702', 'T701', 'T702', 'T703', 'T704', 'T706', 'T708', 'T709', 'T711', 'T712', 'T705', 'FT703', 'FT704', 'PDI701', 'PDI702', 'PY23', 'FYI702']
            arr = df.loc[:seq_len,features].to_numpy().astype(np.float32)
            arr = np.expand_dims(arr, axis=0)           # shape [1, window, features]

            out_np, ms = await run_onnx_inference(arr)
            print("ONNX output shape:", out_np.shape)
            print("ONNX output sample:", out_np.flatten()[:10])
        except Exception as e:
            self.message = f"inference failed — {e}"
            raise

    def detect_anomalies(self, df: pd.DataFrame):
        asyncio.ensure_future(self.apply_onnx_to_df(df))





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





if __name__ == "__main__":
    gui = GUI()
    data_handler = DataHandler()
    data_renderer = DataRenderer()
    anomaly_detector = AnomalyDetector()
    
    scaler = RobustInputScaler()

    gui.file_input.param.watch(data_handler._on_file_change, "value")
    gui.model_input.param.watch(data_handler._on_model_change, "value")

    # new df -> render data
    pn.bind(data_renderer.update_plots, df=data_handler.param.df, watch=True)
    # new plot -> show in gui
    pn.bind(gui._on_update_plots, layout=data_renderer.param.time_series_plot, watch=True)

    # load model
    #pn.bind(anomaly_detector.update_model, model=gui.model_input.param.value, watch=True)
    # message from anomaly detector -> show in gui
    pn.bind(lambda m: gui.model_status.param.update(object=f"**Model**: {m}"), anomaly_detector.param.message, watch=True)
    pn.bind(lambda m: gui.model_status.param.update(object=f"**Model**: {m}"), data_handler.param.message, watch=True)
    # "Detect Anomalies" button pressed
    pn.bind(lambda _clicks: anomaly_detector.detect_anomalies(data_handler.df), gui.start_ad_button.param.clicks, watch=True)

    await pn.io.pyodide.write("sidebar", gui.sidebar)
    await pn.io.pyodide.write("main", gui.main)