# pygraphon

Development still in progress. See below for current plans.

----

Python library for working with graphons.
Uses
* Python 2.7 (Python 3 support coming)
* Numpy
* Scipy

Graphons are symmetric, measurable functions from which we can sample
exchangeable (dense) graphs (the function that appears in the Aldous-Hoover
theorem), and are a graph limit object.

The purpose of this library is to help others [understand graphons better]() and
also further the computational side of working with graphons.


### Install

1. In the top level directory, run the command
```python
python setup.py install
```
You can also run
```python
python setup.py build
```
to build the package.

2. Import the package (or part of it)

```python
import pygraphon
```

Most of the functionality for users will be imports from pygraphon.core.base,
 pygraphon.core.graphons, and (possibly) pygraphon.core.graphon\_utils.

## Basic functionality

A graphon W is a measurable function from [0,1]^2 to [0,1] that is symmetric,
  i.e., W(x,y) = W(y,x).

So, in the codebase, we can pass any function that takes 2 arguments in [0,1]
and returns a single argument in [0,1] for any of the utilities that take in a
"graphon" as an argument; whenever we say "graphon" here, we are referring to the
actual function (rather than a more complicated object).

For example, suppose we have the following graphon:

```python
graphon_ER = lambda x,y: 0.5
```

This corresponds to an Erdos-Renyi graph with probability 0.5.

We can visualize a graphon with a 2-dimensional plot by letting the values of
the axes represented [0,1]^2 and the color at each point representing the value
given by W(x,y). We use a grayscale gradient to represent colors between 0 and 1.


```python
from pygraphon.core.graphon_utils import plot_graphon

plot_graphon(ER)
```

Given any graphon, we can sample an exchangeable graph from it:

```python
from pygraphon.core.graphon_utils import sample_graph
from pygraphon.core.graphon_utils import plot_graph

sample = sample_graph(ER, 100)
plot_graph(sample)
```

Another thing we might want is to take a graph sample (adjacency matrix) and
turn it into a step-function, getting a graphon.

```python
from pygraphon.core.graphon_utils import step_function

stepfn = step_function(sample)
plot_graphon(stepfn)
sample2 = sample_graph(stepfun, 100)
plot_graph(sample2)
```

Alternatively, we can create Graphon and Graph objects and perform the same
operations.
```python
from pygraphon.core.base import Graphon

ER = Graphon(graphon_ER)
sample = ER.sample(100)
ER.plot()
sample.plot()

sample_step = sample.to_stepfunction()
sample_step.plot()
sample2 = sample_step.sample(100)
sample2.plot()
```

In the file graphon\_core/graphons.py, contain a list of implemented graphons.
Some are directly graphons, while others are random and/or have some argument,
e.g., graphons are created after setting some parameter

```python
ERquasi1 = Graphon(ER(0.1))
ERquasi5 = Graphon(ER(0.5))
ERquasi8 = Graphon(ER(0.8))
```

Currently, we have the following graphons/models:
* "gradient"
* Erdos-Renyi [0]
* table of graphons listed in Chan and Airoldi [1]
* 2-parameter block model
* k-parameter stochastic block model [2]
* infinite relational model (CRP) [3]


