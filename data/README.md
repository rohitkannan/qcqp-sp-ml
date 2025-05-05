# Random families of nonconvex QCQPs
Scripts for generating a family of random non-convex quadratically constrained quadratic program (QCQP) instances, as described in Section 5 of the paper.

To generate 1000 random bilinear instances with `N` variables (`N = 10, 20, 50`), run
```
python3 generate_bilinear_instances.py --numVariables N --numInstances 1000
```

To generate 1000 random QCQP instances (including both bilinear and univariate quadratic terms) with `N` variables (`N = 10, 20, 50`), run
```
python3 generate_qcqp_instances.py --numVariables N --numInstances 1000
```

To generate 1000 random pooling instances, run
```
python3 generate_pooling_instances.py --numInstances 1000
```
Additional data files for generating the pooling instances can be found within the `pooling` folder. These data files were generated using the scripts [here](https://github.com/poolinginstances/poolinginstances).