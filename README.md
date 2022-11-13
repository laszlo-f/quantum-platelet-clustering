# quantum-platelet-clustering
Python code that uses a time series and e-divisive changepoints to compute clusters with DBSCAN.

This program reads in a time series and a precomputed energy divisive change point calculation.  The time series is subdivided by change points.  A separation metric for each pair of subdivisions is computed.  The mean of means for each pair of subdivisions is computed.  Then it computes the product of subdivision lengths.  Goal is to create a graphic that visualizes clustering.

This code was created to seek an upper bound on the number of brightness levels in a time series of photoluminescence measurements of a quantum dot.
