fzjax - pure parametrized functions on named trees
--------------------------------------------------
**WORK IN PROGRESS**

This library poses a "pure" alternative to "PyTorch"-like jax libraries/frameworks such as
flax, dm-haiku and equinox, where "Modules" such as `flax.linen.linear` define
parameters as object properties, thus introducing side-effects.

Instead, we define such "Modules" (`pfuncs`) as pure functions, expecting pytrees as input.
As a structured and typed container for `pfunc` parameters, we provide `@fzjax_dataclasses`
which can be used as a native `dataclasses.dataclass`.[^1]


## Named Trees and fzjax_dataclasses
We provide an alternative to `jax.tree_utils.flatten`, which flattens pytrees by name
(i.e. dataclass property names, dictionary keys but also list and tuple indices).
Other classes are treated as leaves, unless registered separately.

For example:

```python
@fzjax_dataclass
class Params:
    lr: Meta[int] = .1
    weights: Differentiable[tuple[jnp.ndarray]] = (w1, w2)
    states: jnp.ndarray = s1
p = Params()
```
is flattened as 
```python
{
    "lr": FlattenLeaf(.1, (JDC_META_MARKER,)),
    "weights.0": FlattenLeaf(w1, (JDC_DIFF_MARKER,)),
    "weights.1": FlattenLeaf(w1, (JDC_DIFF_MARKER,)),
    "states": FlattenLeaf(s1, tuple())
}
```

This facilitates generating and modifying subsets of treedefs. 

### Annotations
Annotations are passed onto every child node.

The `Meta` annotation moves parameters into the treedef (equivalent to `jdc.Static` [^1]).
The `Differentiable` annotation marks parameters for the `named_tree_differentiable` function,
e.g.
```python
named_tree_differential(p) = {
    "weights.0": w1,
    "weights.1": w2
}
```

### Higher Functions
This alternative formulation now allows computing the gradient of a function **w.r.t. parts of a PyTree**).
For now, we have implemented `pfunc_value_and_grad`, which could be used as follows
```python
def bilinear(p: Params, x: jnp.ndarray) -> jnp.ndarray:
    ...

vg = pfunc_value_and_grad(bilinear, ["weights.0"])
value, grad = vg(p)
# grad: {"weights.0": jnp.ndarray}
```


### IDEs and fzjax_dataclasses
Since PyCharm to this date still does not fully implement [PEP 681](https://peps.python.org/pep-0681/)[^2],
a workaround is to define fzjax_dataclasses as:

```python
@fzjax_dataclass
@dataclass
class Params:
    ...
```

[^1] Shamelessly stolen and adapted from the great [jax_dataclasses](https://github.com/brentyi/jax_dataclasses) library.
[^2] https://youtrack.jetbrains.com/issue/PY-54560