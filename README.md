# Similarity Search of Low Surface Brightness Galaxies in Large Astronomical Catalogs

A PySpark-based Locality-Sensitive Hashing (LSH) model applied to the Dark Energy Survey Y3 Gold coadd catalog to perform an approximate similarity search for Low Surface Brightness Galaxies.

## About

Low Surface Brightness Galaxies (LSBGs) constitute an important segment of the galaxy population, however, due to their diffuse nature, their search is challenging. The detection of LSBGs is usually done with a combination of parametric methods and visual inspection, which becomes unfeasible for future astronomical surveys that will collect petabytes of data. Thus, in this work we explore the usage of Locality-Sensitive Hashing for the approximate similarity search of LSBGs in large astronomical catalogs. We use 11670190 objects from the Dark Energy Survey Y3 Gold coadd catalog to create an approximate $k$ nearest neighbors model based on the properties of the objects, developing a tool able to find new LSBG candidates while using only one known LSBG. From just one labeled example we are able to find various  known LSBGs and many objects visually similar to LSBGs but not yet catalogued. Also, due to the generality of similarity search models, we are able to search for and recover other rare astronomical objects without the need of retraining or generating a large sample.

# Code

Our LSH pipeline is available on `lsh.py`, and it contains the code we used to load & process the data as well as the code we used for our model and for our tests. In it we search for both LSBGs and artifacts (objects that are similar to LSBGs, but aren't considered LSBGs).

# Data

For our data we use the [Dark Energy Survey Y3 Gold coadd catalog](https://des.ncsa.illinois.edu/releases/y3a2/Y3gold) to train our model and perform our searches. The query we used to gather the data on [DESacces](https://des.ncsa.illinois.edu/desaccess/) is available on `query.sql`.

We used [Tanoglidis et al.'s LSBG catalog](https://iopscience.iop.org/article/10.3847/1538-4365/abca89) for our LSBG keys and [Tanoglidis et al.'s LSBG artifact catalog](https://www.sciencedirect.com/science/article/pii/S2213133721000238?via%3Dihub) for our artifact keys.

# Results

We obtain our results by performing searches with random keys (on `lsh.py`). We used the code available on `results_viz.py` to create our visualizations. The `figs/` directory contains the visualizations for our other keys.  

