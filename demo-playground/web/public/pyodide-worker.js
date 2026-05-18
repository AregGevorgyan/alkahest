// Pyodide Web Worker — runs SymPy and pure-Python cells in the browser
// Alkahest cells (import alkahest) are always routed to the server instead.

importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js');

let pyodide = null;

async function init() {
  pyodide = await loadPyodide();

  // Install common scientific packages
  await pyodide.loadPackage(['numpy', 'sympy', 'micropip']);

  // Redirect stdout/stderr capture helpers
  pyodide.runPython(`
import sys, io, base64, json

class _Capture:
    def __init__(self, name):
        self.name = name
        self._buf = []
    def write(self, s):
        self._buf.append(s)
    def flush(self):
        pass
    def getvalue(self):
        return ''.join(self._buf)
    def clear(self):
        self._buf.clear()

_stdout_cap = _Capture('stdout')
_stderr_cap = _Capture('stderr')

def _reset_capture():
    _stdout_cap.clear()
    _stderr_cap.clear()
    sys.stdout = _stdout_cap
    sys.stderr = _stderr_cap

def _restore_streams():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Detect LaTeX in text: lines wrapped in $$ or $ ... $
import re as _re
_LATEX_BLOCK = _re.compile(r'^\$\$(.+?)\$\$', _re.DOTALL)
_LATEX_INLINE = _re.compile(r'^\$(.+?)\$')

def _classify_output(text):
    text = text.strip()
    if _LATEX_BLOCK.match(text):
        return ('latex', text[2:-2].strip())
    if _LATEX_INLINE.match(text) and '\\\\' in text:
        return ('latex', text[1:-1].strip())
    return ('text', text)
`);

  self.postMessage({ type: 'ready' });
}

async function execute(id, code) {
  const outputs = [];

  try {
    pyodide.runPython('_reset_capture()');

    // Capture display() calls from matplotlib / IPython-style display
    pyodide.globals.set('_worker_outputs', outputs);
    pyodide.runPython(`
import builtins as _bi

_orig_display = _bi.__dict__.get('display', None)

def _display_hook(*args, **kwargs):
    import io, base64
    for obj in args:
        # Handle matplotlib figures
        try:
            import matplotlib.pyplot as plt
            if hasattr(obj, 'savefig') or isinstance(obj, plt.Figure):
                buf = io.BytesIO()
                obj.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                data = base64.b64encode(buf.read()).decode()
                _worker_outputs.append({'type': 'image', 'format': 'png', 'data': data})
                return
        except Exception:
            pass

        # Handle objects with _repr_latex_
        if hasattr(obj, '_repr_latex_'):
            latex = obj._repr_latex_()
            if latex:
                _worker_outputs.append({'type': 'latex', 'latex': latex})
                return

        # Fallback to repr
        _worker_outputs.append({'type': 'text', 'stream': 'stdout', 'text': repr(obj)})

_bi.display = _display_hook
`);

    // Run the user code
    await pyodide.runPythonAsync(code);

    // Flush matplotlib figures that weren't explicitly displayed
    pyodide.runPython(`
try:
    import matplotlib.pyplot as plt
    import io, base64
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode()
        _worker_outputs.append({'type': 'image', 'format': 'png', 'data': data})
    plt.close('all')
except Exception:
    pass
`);

    // Collect stdout/stderr
    const stdout = pyodide.runPython('_stdout_cap.getvalue()');
    const stderr = pyodide.runPython('_stderr_cap.getvalue()');

    if (stdout) {
      // Classify each line
      for (const line of stdout.split('\n')) {
        if (!line.trim()) continue;
        const [kind, content] = pyodide.runPython(`_classify_output(${JSON.stringify(line)})`).toJs();
        outputs.push(kind === 'latex' ? { type: 'latex', latex: content } : { type: 'text', stream: 'stdout', text: line });
      }
    }
    if (stderr) {
      outputs.push({ type: 'text', stream: 'stderr', text: stderr });
    }

    pyodide.runPython('_restore_streams()');

    self.postMessage({ type: 'result', id, outputs });
  } catch (err) {
    const stderr = (() => {
      try { return pyodide.runPython('_stderr_cap.getvalue()'); } catch { return ''; }
    })();
    try { pyodide.runPython('_restore_streams()'); } catch {}

    outputs.push({ type: 'error', ename: 'PythonError', evalue: err.message, traceback: [stderr || err.message] });
    self.postMessage({ type: 'result', id, outputs });
  }
}

self.onmessage = async (event) => {
  const { type, id, code } = event.data;
  if (type === 'execute') await execute(id, code);
};

init().catch((err) => {
  self.postMessage({ type: 'error', id: '', error: `Pyodide init failed: ${err.message}` });
});
