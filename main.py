# main.py
import panel as pn
from panel.template import FastListTemplate

# Make sure Panel extensions loaded
pn.extension(sizing_mode="stretch_width")

# widgets / content
slider = pn.widgets.FloatSlider(name="Amplitude", start=0, end=10, value=5)
text = pn.bind(lambda v: f"Amplitude is: {v:.2f}", slider)
main_content = pn.Column(
    pn.pane.Markdown("## My FastListTemplate App"),
    pn.panel(text)
)

# Build the template
template = FastListTemplate(
    title="Demo FastListTemplate",
    sidebar=[pn.pane.Markdown("### Controls"), slider],
    main=[main_content]
)

# Explicitly write the template into the DOM element with id="app"
# In a Pyodide context we await this call so the frontend receives the view.
await pn.io.pyodide.write("app", template)
