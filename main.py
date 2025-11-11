import panel as pn
pn.extension(sizing_mode="stretch_width")

slider = pn.widgets.FloatSlider(start=0, end=10, name="Amplitude")
def cb(v):
    return f"Amplitude is: {v}"

pn.Row(slider, pn.bind(cb, slider)).servable(target="app")
# or: await pn.io.pyodide.write('app', pn.Row(...))  if you prefer explicit write
