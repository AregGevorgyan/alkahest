# Interoperability

Alkahest integrates with the Python numerical ecosystem at well-defined boundaries.

## NumPy

### Batch evaluation

`numpy_eval` vectorises a compiled function over NumPy arrays with zero unnecessary copies:

```python
import numpy as np
from alkahest import ExprPool, compile_expr, numpy_eval, sin

pool = ExprPool()
x = pool.symbol("x")

f = compile_expr(sin(x) ** 2 + x, [x])
xs = np.linspace(0, 2 * np.pi, 1_000_000)
ys = numpy_eval(f, xs)   # returns a NumPy array, shape (1_000_000,)
```

Inputs are converted to `f64` arrays via DLPack or `__array__`. The call is vectorised through `call_batch_raw` in Rust — no Python loop.

### Array protocol

`CompiledFn` objects implement `__array__` for direct NumPy coercion:

```python
result = np.asarray(f([1.0]))  # scalar result as a 0-d array
```

## PyTorch

PyTorch CPU tensors are accepted wherever NumPy arrays are (via `__dlpack__`):

```python
import torch
xs = torch.linspace(0, 1, 10_000)
ys = numpy_eval(f, xs)   # returns a NumPy array
```

For GPU tensors, use the `compile_cuda` path (requires `--features cuda`), which accepts device pointers via `call_device_ptrs`.

## JAX

### numpy_eval with JAX arrays

JAX arrays implement `__dlpack__` and are accepted by `numpy_eval`:

```python
import jax.numpy as jnp
xs = jnp.linspace(0, 1, 10_000)
ys = numpy_eval(f, xs)
```

### JAX primitive source (to_jax)

`to_jax` registers a symbolic expression as a JAX primitive, making it callable inside JAX computations including `jax.jit`, `jax.grad`, and `jax.vmap`:

```python
from alkahest import to_jax, ExprPool, sin

pool = ExprPool()
x = pool.symbol("x")

jax_fn = to_jax(sin(x) ** 2, [x])

import jax
import jax.numpy as jnp

# Use inside jax.jit / jax.grad
jit_fn = jax.jit(jax_fn)
grad_fn = jax.grad(lambda x: jax_fn(x).sum())
```

The primitive registers:
- A concrete `def_impl` that calls the Rust evaluator
- An abstract evaluation rule for shape/dtype propagation
- A JVP (forward-mode) rule derived from the symbolic gradient
- A vmap batching rule

### StableHLO / XLA

`to_stablehlo` emits textual MLIR in the StableHLO dialect, which XLA and JAX's XLA backend can compile:

```python
from alkahest import to_stablehlo

mlir_text = to_stablehlo(expr, [x, y], fn_name="my_kernel")
# Pass to xla_client.compile() or save to .mlir file
```

## SymPy interop

Alkahest does not import SymPy at runtime. The integration is one-way for validation: the test oracle in `tests/test_oracle.py` uses SymPy as a ground truth reference. The recommended pattern for mixed workflows is to convert to/from string representation.

## DLPack

All DLPack-compatible arrays (NumPy, PyTorch, JAX, CuPy) are accepted at the `numpy_eval` and `call_device_ptrs` boundaries. The DLPack conversion is zero-copy for CPU arrays with matching dtypes.

## Exporting C code

`emit_c` generates a standalone C function for embedding in other projects:

```python
from alkahest import emit_c

c_code = emit_c(
    sin(x) * exp(pool.integer(-1) * x),
    [x],
    var_name="x",
    fn_name="damped_sin",
)
print(c_code)
# double damped_sin(double x) { return sin(x) * exp(-x); }
```

The emitted code uses only standard `<math.h>` functions and has no Alkahest dependency.
