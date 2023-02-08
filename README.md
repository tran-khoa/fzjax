fzjax - pure parametrized functions on trees with paths
-------------------------------------------------------
**WORK IN PROGRESS, API likely to change**

This library poses a "pure" alternative to "PyTorch"-like jax libraries/frameworks such as
flax, dm-haiku and equinox, where "Modules" such as `flax.linen.linear` define
parameters as object properties, thus introducing side-effects.

Instead, we define such "Modules" (`pfuncs`) as pure functions, expecting pytrees with paths (ptrees) as input.
As a structured and typed container for `pfunc` parameters, we provide `@fzjax_dataclasses`
which can be used as a native `dataclasses.dataclass`.[^1]


## ptrees and fzjax_dataclasses
We provide an alternative to `jax.tree_utils.flatten`, which flattens pytrees by path
(i.e. dataclass property names, dictionary keys but also list and tuple indices).
The implementation is based on `chex.dataclasses` and `dm-tree`.

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
# ptree_flatten(p)
{
    "lr": FlattenLeaf(.1, (JDC_META_MARKER,)),
    "weights.0": FlattenLeaf(w1, (JDC_DIFF_MARKER,)),
    "weights.1": FlattenLeaf(w1, (JDC_DIFF_MARKER,)),
    "states": FlattenLeaf(s1, ())
}
```

This facilitates generating and modifying subsets of trees. 

### Annotations
Annotations are passed onto every child node.

The `Meta` annotation moves parameters into the treedef (equivalent to `jdc.Static` [^1]).
The `Differentiable` annotation marks parameters for the `ptree_differentiable` function,
e.g.
```python
ptree_differentiable(p) = {
    "weights.0": w1,
    "weights.1": w2
}
```
The `NoDiff` annotation marks itself and its children as non-differentiable, overriding
`Differentiable` annotations.

### Higher Functions
This alternative formulation now allows computing the gradient of a function **w.r.t. parts of a PyTree**).
For now, we have implemented `pfunc_value_and_grad` and `pfunc_jit`, which could be used as follows
```python
@pfunc_jit
def bilinear(weights: tuple[jnp.ndarray, jnp.ndarray], x: Donate[jnp.ndarray]) -> jnp.ndarray:
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

[^1]: Shamelessly stolen and adapted from the great [jax_dataclasses](https://github.com/brentyi/jax_dataclasses) library.
[^2]: https://youtrack.jetbrains.com/issue/PY-54560
