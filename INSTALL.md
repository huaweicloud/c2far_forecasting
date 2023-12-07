### Environment

We conducted experiments in the `c2far` env.  

You can set it up with:

```
conda env create --name c2far --file conda_environment.yml
```

The file `conda_environment.yml` was generated with:

```
conda env export > conda_environment.yml
```

The python version was 3.7.11, and the main dependencies are numpy,
matplotlib, and torch, all installed through pip.
