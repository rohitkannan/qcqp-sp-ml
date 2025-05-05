[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex QCQPs

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under a [License from Triad National Security, LLC](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported [in the paper](https://doi.org/10.1287/ijoc.2023.0424) by R. Kannan, H. Nagarajan, and D. Deka. 


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

[The journal article](https://doi.org/10.1287/ijoc.2023.0424):

```bibtex
@article{kannan2025strong,
  author = {Kannan, Rohit and Nagarajan, Harsha and Deka, Deepjyoti},
  title = {Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex {QCQPs}},
  journal = {INFORMS Journal on Computing},
  year = {2025},
  doi = {10.1287/ijoc.2023.0424},
}
```

[This repository](https://doi.org/10.1287/ijoc.2023.0424.cd):

```bibtex
@misc{kannan2025github,
  author =        {Kannan, Rohit and Nagarajan, Harsha and Deka, Deepjyoti},
  publisher =     {INFORMS Journal on Computing},
  title =         {Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex {QCQPs}},
  year =          {2025},
  doi =           {10.1287/ijoc.2023.0424.cd},
  url =           {https://github.com/INFORMSJoC/2023.0424},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0424},
}  
```

## Description

The goal of this software is to demonstrate the effect of cache optimization.

## Building

In Linux, to build the version that multiplies all elements of a vector by a
constant (used to obtain the results in [Figure 1](results/mult-test.png) in the
paper), stepping K elements at a time, execute the following commands.

```
make mult
```

Alternatively, to build the version that sums the elements of a vector (used
to obtain the results [Figure 2](results/sum-test.png) in the paper), stepping K
elements at a time, do the following.

```
make clean
make sum
```

Be sure to make clean before building a different version of the code.

## Results

Figure 1 in the paper shows the results of the multiplication test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/mult-test.png)

Figure 2 in the paper shows the results of the sum test with different
values of K using `gcc` 7.5 on an Ubuntu Linux box.

![Figure 1](results/sum-test.png)

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
