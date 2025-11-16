# panel-pyodide-test
Test panel in pyodide


## Long loading times for pyodide

Download using 
```
cd static
curl -L https://github.com/pyodide/pyodide/releases/download/0.28.2/pyodide-0.28.2.tar.bz2 -o pyodide.tar.bz2
tar -xjf pyodide.tar.bz2
rm pyodide.tar.bz2
cd ..
```

with online version:
- add line in head <script type="text/javascript" src="https://cdn.jsdelivr.net/pyodide/v0.28.2/full/pyodide.js"></script>
- use normal load: const pyodide = await loadPyodide();

for downloaded version
- uncomment all from above
- import local verison: import { loadPyodide } from "/static/pyodide/pyodide.mjs";
- use special load
- make the script a module: <script type="module">


> python serve.py

## using files in src
```
cd src && zip -r ../static/src.zip * && cd ..
```


## uploading model

zip the trained model and the two scalers (input + target). call it model.zip. Upload under model.