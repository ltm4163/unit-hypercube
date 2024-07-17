# Classification in High Dimensional Unit Space

<p>The purpose of this project is to explore the effectiveness of different classification models in higher dimensional unit space. 
</p>

# The Code

<p>The project includes the following files:
</p>

<ol>
    <li> <strong>DataGenerator.py</strong> includes functions for generating sample datasets in the unit space</li>
    <li> <strong>hypercube.ipynb</strong> trains, visualizes, and analyzes different models on generated data</li>
    <li> <strong>test</strong> is a directory containing example unit space data to test with</li>
    <li> <strong>train</strong> is a directory containing example unit space data to train with</li>
</ol>

# The Data

The generated example data is intended to model real world situations that may occur in the unit space. The data generation is designed to be easily extendable into higher dimensions. Three static train/test data sets are included in the train/test directories. The motivation behind each distribution is detailed below.

<ol>
    <li> <strong> Patient Vital Signs </strong> the first distribution is intended to replicate what vital sign monitoring data might look like in the unit space. </li>
    <li> <strong> Traffic Flow </strong> the second distribution is meant to represent traffic flow from different sensors, and takes a linear shape in 2d.
    <li> <strong> Political Ratings </strong> the third distribution models political ratings, and takes a sinusoidal shape in 2d </li>
</ol>

Each of these situations is meant to represent a real world situation that would produce data in the unit space. 
