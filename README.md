# detectVirus

Example programm in sci-kit's web that compares algorithm's performances (http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#sphx-glr-auto-examples-linear-model-plot-sgd-comparison-py). Slightly modified so the input data is our own training dataset. This programm uses no test dataset. Splits the training database in order to use a part as a trainer and the other part as test data. Pending of modifies so it uses the whole database to train and then picks a test dataset and classifies it


# Preprocessing
The TxtProcessing contains a Java project that transforms the datasets ".txt" files into ".csv" in a way that allows us to process the data.


The first and last numbers shows us if the file is malicious or not, being:

**+1 ... -1** Non-malicious

**-1 ... -1** Malicious

The numbers in between indicates us which features exist in the file, if a feature is omitted it means it does not exist:

**20:1** Feature 20 exists

Features go from 1 to 531


The original format it's:

**+1 1:1 5:1 ... 531:1 -1**

After processing it:

**0 1 0 0 0 1 ... 1**

Indicating the first column whether if it is malicious **(1)** or not **(0)** and the following numbers the existance of each of the features from 1 to 531
