# Same Stats, Different Graphs: An Instant Transformation Method

https://diglib.eg.org/server/api/core/bitstreams/ed5978ff-a171-46ae-a184-a82602930c44/content
# Environment instructions

Python version 3.10.4

Install packages with: 'pip install -r requirements.txt'. 

## Recreating target shapes with a desired set of summary statistics
1. To transform an existing dataset with a desired shape into a dataset with a target set of desired summary statistics (using the proposed linear transformation method) run **main_fool.py**.
- 'python main_fool.py --help' to see the arguments
- e.g. 'python main_fool.py -t simple_tree_850 -xm 60 -ym -xsd -ysd -pc 0.1'
- The above example will take the input dataset (simple_tree_850.csv located in target_datasets/) and morph it to have an X mean of 60, the default Y mean value (47.835), the default X standard deviation value (16.765), the default Y standard deviation value (26.935) and a pearson correlation of 0.1. You can experiment with these values.
2. (Optional): To create a scatterplot and a dataset from an existing image run **pic_to_dataset.py**.
- 'python pic_to_dataset.py --help' to see the arguments
- e.g. 'python pic_to_dataset.py -img simple_tree.png -n 850'
- The above example will take the input image (simple_tree.png located in target_datasets/input_images/) and turn it into a point cloud of 850 points. Note that this script is only tested for black and white images, where the background is white and a black pen is used.
- The newly created scatterplot and dataset will be saved in /target_datasets/ and can now be used using **main_fool.py**.


## Recreating exact experimental conditions

To recreate the experiments performed in the paper:
1. Select the method for recreating target shapes:
- **project_implem.py**: the mathematical approach to acquire a specified covariance matrix, a non-iterative fast approach with good results.
- **datasaurus_dozen_implem.py**: the Datasaurus Dozen Simulated Annealing approach, slow and iterative.
- **datadance_doubledozen_implem.py**: the more precise Simulated Annealing approach, from the DataDance DoubleDozen. Slow and iterative but more exact in shape replication.
- **sgd_implem.py**: a fast gradient descent approach, fast and iterative and uses the mathematical approach from **project_implem.py** to acquire the target covariance matrix.
2. In the chosen script, adjust the variables in the main function to the target shape that you want to replicate and the target summary statistics. You can also adjust the starting shape for all scripts (except for **project_implem.py**). Do note that the starting shape has a fixed set of summary statistics. One can use the function **create_random_cloud** in **project_implem.py** to create a random cloud of 142 points with specified summary statistics.


## DataDance DoubleDozen videos

# Proposed Linear Transformation method results:
<img src="dance_gifs/project_implem-datadance250_stats.gif" width="425"/> 

# SGD results:
<img src="dance_gifs/sgd_implem-datadance250_stats.gif" width="425"/>

# Datasaurus Dozen method results:
<img src="dance_gifs/dd_implem-datadance250_stats.gif" width="425"/>

# DataDance DoubleDozen method results:
<img src="dance_gifs/dddd_implem-datadance250_stats.gif" width="425"/>