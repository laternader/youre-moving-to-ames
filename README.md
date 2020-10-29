# Let's Find Timmy a Home

*If you are moving down to Ames, I am sure you want to know if you are paying the right price for the type of house you want. Whether the house has a pool, a missing garage, or has a veneer the size of a queen sized bed, you want to know whether or not the price for that house is the right price. Remember in this economy, every penny counts. It is my job to look out for you and prevent and real estate agent from cheating you out of a good deal.*

**Problem Statement**:
My friend, Timmy, is moving to Ames, IA (maybe because of their low taxes over there). I am tasked with finding out how much a house should cost given a set of features on specific house. My friend was kind enough to "get a hold of" data pertaining to the cost of a house in Ames. However, it is incomplete! Some of the sale prices are missing! He doesn't want to make a decision on a house yet until he has a good idea how much every house costs. He also wants to know if certain features of a home are worth paying the price. Since you are a kind friend (and because he is paying you good money), you use Linear Regression and other models to predict the price of a home. Who knows maybe this data can be presented to a larger audience interested in your results?

![Timmy](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/imgs/timmy.jpg)

## Directory
- [EDA](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/1%20-%20EDA.ipynb)
- [Feature Selection Notebook](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/1.5%20-%20Select-Features.ipynb)
  - This archived all my feature selections.
- Cleaning/Mapping Notebooks
  - [Train Dataset](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/2%20-%20Train-Cleaning.ipynb)
  - [Test Dataset](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/2%20-%20Test-Cleaning.ipynb)
- [Feature Engineering](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/3%20-%20Feature-Engineering.ipynb)
- [Modeling Benchmarks](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/4%20-%20Modeling-Benchmarks.ipynb)
  - [Modeling Benchmarks Logged](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/4.5%20-%20Log-Benchmarks.ipynb)
- [Submission Maker](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/code/5%20-%20Submission-Maker.ipynb)
- [Data Folders](https://git.generalassemb.ly/laternader/project_2/tree/master/deliverables/data)
- [Submission Folders](https://git.generalassemb.ly/laternader/project_2/tree/master/deliverables/submissions)
- [Image Folders](https://git.generalassemb.ly/laternader/project_2/tree/master/deliverables/imgs)

# Additional Information/Research
There's a good number of sleazy tactics a real estate agents can pull on a first time homeowner or a first time home-seller. In fact, one of the few tactics that real estate agents do is "price a home on something other than its true value" according to this [article](https://blog.duvora.com/top-10-dirty-tricks-used-by-real-estate-agents/). As mortgages rise (or fall), depending on location, the price of a home is never set in stone.

# Dataset
The original data was split into two sets: `train` and `test`. Within these two files contained about 81 columns/features worth of data. The `test` data however was missing the "sale price" column. I was given the task of calculating the price of a home in the `test` data set based on their features. I had to map a lot of missing values either to another category, the mean of the entire feature set, or fill them with zeroes. I am given the freedom to create my model based on the features I select. I can use Linear Regression, Lasso, Ridge, and other different types of models to better my chances as well as improve my accuracy on predicting the price of a home. I properly fitted my models using split data based on the `train` data set in order to prep for predicting sale price for the homes in the `test` data frame. I am also given the freedom to engineer any features, or interactions, that can influence the model. I also need to show any trends for any houses with particular features such as central air or basement conditions in order to fully understand the price of a home.

### Data Dictionary:
|**Feature**|**Description**|**Type**|
|---|---|---|
|'ms_subclass'|The building class ranging from 1 stories, 1.5 stories, 2 stories, 2.5 stories, multi-levels, split foyers, duplexes, PUDs, and family style homes|*int*|
|'street'|Type of road access to property|*object*|
|'alley'|Type of alley access to property|*object*|
|'neighborhood'|Physical locations within Ames city limits|*object*|
|'bldg_type'|Type of dwelling|*object*|
|'house_style'|Style of dwelling|*object*|
|'overall_qual'|Overall material and finish quality of the home|*int*|
|'overall_cond'|Overall condition rating of the home|*int*|
|'year_built'|Original construction date of the home|*int*|
|'year_remod/add'|Remodel date (same as construction date if no remodeling was done)|*int*|
|'bsmt_qual'|Height of basement given as a rating|*object*|
|'bsmt_cond'|General condition of the basement|*object*|
|'total_bsmt_sf'|Total square feet of the basement area|*int*|
|'heating_qc'|Heating quality gauged by a rating system|*object*|
|'central_air'|Central Air conditioning? Yes or no.|*object/uint*|
|'electrical'|Electrical system of the home|*object*|
|'1st_flr_sf'|First floor square feet|*int*|
|'gr_liv_area'|Above grade/ground living area in square feet|*int*|
|'full_bath'|Number of full bathrooms above grade|*int*|
|'half_bath'|Number of half bathrooms above grade|*int*|
|'bedroom_abvgr'|Number of bedrooms above basement level|*int*|
|'kitchen_qual'|Kitchen quality based on rating system|*object*|
|'totrms_abvgrd'|Total rooms above grade (does not include bathrooms)|*int*|
|'functional'|Home functionality rating. Is the house livable?|*object*|
|'fireplaces'|Number of fireplaces|*int*|
|'garage_yr_blt'|Year garage was built (if applicable)|*int*|
|'garage_cars'|Size of garage in car capacity|*int*|
|'garage_area'|Size of garage in square feet|*int*|
|'garage_qual'|Garage quality|*object*|
|'garage_cond'|Condition of the garage|*object*|
|'mo_sold'|Month sold|*int*|
|'yr_sold'|Year sold|*int*|
|'sale_type'|Type of sale for the house|*object*|
|'mas_vnr_area'|Masonry veneer area in square feet|*int*|

# Process and Discussion
I checked the distribution of the prices of a home and noticed that there was a right skew distribution.

![Distribution of SalePrice](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/imgs/hist-saleprice.png)

The first thing was select my first set of features. I decided that I would take the correlation of all the numeric values and compare them to the sale price. I then selected the features that had a correlation above .5 since correlation tends to mean that as one unit increases in one feature, the price increases by x amount. For the first set I selected these features:
+ Masonry Veneer area
+ Number of full bathrooms
+ Year the house was built and, if applicable, remodeled
+ First floor square foot area
+ Total basement square foot area
+ Above ground living area in square feet
+ Garage square foot area, car capacity, year built
+ Overall quality of the home

![Correlation Heatmap Sale Price](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/imgs/heatmap-saleprce.png)

This was the starting point. After fitting the model to a Linear Regression, LassoCV, Ridge, and RidgeCV model, I determined that the Linear Regression model gave the best results in low Root Mean Squared Error (RMSE). The smaller RMSE is then the smaller our error is from predicting the price of a home. That result was about $27,000. The goal was to get it as small as possible.

Certain neighborhoods were under-represented in the data that figuring out the price of a home based on location was difficult. However, it gives me an idea of where my friend can move to.

![Neighborhoods](https://git.generalassemb.ly/laternader/project_2/blob/master/deliverables/imgs/wheredaneighbors.png)

As I went further into the project, I produced the best model I could. In this model, I took the log of the sale price in the train data, began to fit the splits into a Linear Regression. Then I predicted on the test data and exponentiated the predictors. That was able to produce the best RMSE at about ~24k.

# Conclusion
If given more time, I would done a much more thorough job at selecting features. I also needed to improve on variety of models used. I wanted to account for categorical variables in the `LassoCV`, `Ridge`, and `RidgeCV` models but I was having trouble appending dummy variable columns to columns that have been split and standardized. I also wanted to include more interactions as features because I felt like there was a decent relationship between bedrooms and bathrooms, just to name a few.

In the end, I recommend that my friend Timmy consider a place with a fireplace and not in the Northridge Heights or Stonebrook neighborhoods. He shouldn't have to worry about the number of rooms a house has because it seems to lower the value by a little bit so then he can rent out the rooms to help pay for the house. He should avoid places with garages, basements, a grandiose kitchen, and contracts that tell him to pay a 15% down-payment. These features are considered "bonuses" if he can afford them. I wish I could select a house that matches these exactly however, I do have any data on houses that need to be sold. I think it would be great to consider taxes and weather patterns in the area as they help determine the price of the home. Being located in Iowa, there are many variables to consider outside of the ones provided. Timmy needs a good realtor that will accept his analysis and give him a good deal on a home.
