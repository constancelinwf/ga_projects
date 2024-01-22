# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Singapore Housing Modelling


## 1. Introduction
The Housing Development Board (HDB) was setup to tackle the housing crisis in 1960s and now serves as the sole agency for public housing. 
HDB homes more than 80% of Singapore residents,  across 24 towns, grouped into 5 areas - North, North-East, East, Central, West. The towns are further grouped into mature or non-mature estates. Mature estates are usually at more central locations with more amenities - demands for residents wanting to own a unit in these areas are higher. Conversely for non-mature estates, the locations are less central with lesser amenities and general demand is lower. 

The units offered by HDB have a wide range of flat types (2-room, 3-room, 4-room, 5-room, executive) to suit the space and budgeting needs of different residents. In addition, there is a  variety of features depending on flat type. Key features of HDB’s public housing policy includes the sale of public houses to residents, but with a limited 99 year lease and it can be bought back by the government. 


## 2. Problem Statement of Project
As property buyers in Singapore, the affordability of a HDB unit and potential factors that can influence the price are of interest because this can guide them in their planning process before making a purchase.
We make use of a regression model to give insights to prospective buyers on factors relating to the HDB unit that affect resale prices.


## 3. Datasets
The Singapore Housing Dataset contains historical data related to resale transactions for Housing Development Board (HDB) flats that took place between the period of March 2012 to April 2021.
The following data sets will be used for the project obtained from [Kaggle](https://www.kaggle.com/competitions/dsi-sg-project-2-regression-challenge-hdb-price/data).
1. train.csv
    - contains variables ranging from housing specifications to amenities for 150,634 HDB flats from 2012 to 2021
2. test.csv
    - similar to train.csv, excluding the resale prices of the HDB 
    - contain details for 16,737 HDB  flats from 2012 to 2021


 ### Data dictionary:
|Feature|Type|Source|Dataset|Description|
|---|---|---|---|---|
|id|object|original dataset|train.csv, test.csv|shows ID of the transaction| 
|town_region|object|-|-|feature engineered by recategorizing "town" column - HDB township where the flat is located| 
|flat_type|object|original dataset|train.csv, test.csv|type of the resale flat unit| 
|flat_model|object|original dataset|train.csv, test.csv| HDB model of the resale flat recategorized into broader categories| 
|tranc_month|object|original dataset|train.csv, test.csv|month of resale transaction| 
|||||
|hdb_age_when_sold|int64|-|-|feature engineered by subtracting lease_commence_date from tranc_year|
|dwelling_density|float64|-|-|feature engineered by dividing total_dwelling_units by max_floor_lvl|
|resale_price|float64|original dataset|train.csv|the property's sale price in Singapore dollars|
|tranc_year|int64|original dataset|train.csv, test.csv|year of resale transaction|
|mid_storey|int64|original dataset|train.csv, test.csv|median value of storey_range|
|floor_area_sqft|float64|original dataset|train.csv, test.csv|floor area of the resale flat unit in square feet|
|mall_nearest_distance|float64|original dataset|train.csv, test.csv|distance (in metres) to the nearest mall|
|mall_within_2km|float64|original dataset|train.csv, test.csv|number of malls within 2 kilometres|
|hawker_nearest_distance|float64|original dataset|train.csv, test.csv|distance (in metres) to the nearest hawker centre|
|hawker_within_2km|float64|original dataset|train.csv, test.csv|number of hawker centres within 2 kilometres|
|hawker_food_stalls|int64|original dataset|train.csv, test.csv|number of hawker food stalls in the nearest hawker centre|
|hawker_market_stalls|int64|original dataset|train.csv, test.csv|number of hawker and market stalls in the nearest hawker centre|
|mrt_nearest_distance|float64|original dataset|train.csv, test.csv|distance (in metres) to the nearest MRT station|
|bus_stop_nearest_distance|float64|original dataset|train.csv, test.csv|distance (in metres) to the nearest bus stop| 
|pri_sch_nearest_distance|float64|original dataset|train.csv, test.csv|distance (in metres) to the nearest primary school|
|vacancy|int64|original dataset|train.csv, test.csv|number of vacancies in the nearest primary school|
|sec_sch_nearest_dist|float64|original dataset|train.csv, test.csv|distance (in metres) to the nearest secondary school|
|cutoff_point|int64|original dataset|train.csv, test.csv|PSLE cutoff point of the nearest secondary school|
|||||
|rental_units_in_blk|int32|-|-|feature engineered for boolean value if there is rental units in the same block
|commercial|int64|original dataset|train.csv, test.csv|boolean value if resale flat has commercial units in the same block|
|multistorey_carpark|int64|original dataset|train.csv, test.csv|boolean value if resale flat has a multistorey carpark in the same block|
|precinct_pavilion|int64|original dataset|train.csv, test.csv|boolean value if resale flat has a pavilion in the same block|
|bus_interchange|int64|original dataset|train.csv, test.csv|boolean value if the nearest MRT station is also a bus interchange|
|mrt_interchange|int64|original dataset|train.csv, test.csv|boolean value if the nearest MRT station is a train interchange station|
|pri_sch_affiliation|int64|original dataset|train.csv, test.csv|boolean value if the nearest primary school has a secondary school affiliation|
|affiliation|int64|original dataset|train.csv, test.csv|boolean value if the nearest secondary school has an primary school affiliation|


## 4. Summary of Analysis
### (a) Model Selection
**Ridge regularization** is chosen to be the model of choice.

Using Linear Regression as the baseline, even though it seems that Lasso regularization will be the optimal choice as it has produced better R2 scores on both the training and test sets as well as K-fold cross validation scores, it has the highest RMSE value out of the 3 models.

On the other hand, even though Ridge regularization has slightly higher R2 scores, it still performed better than linear regression. The train and test score difference is considered minimal - it means that the model still trained sufficiently well and there is only a very slight overfit on the test data. Most importantly, out of the 3 models, Ridge regularization has the lowest RMSE score - this is important as we would want a lower score so that the predicted resale prices would have a lower margin of error compared to the actual resale prices.

### (b) Results from model
The top 3 features that influence resale prices are:
- town region North
- town region West
- town region North-east


## 5. Recommendations
Based on our problem statement, we came up with recommendations for 3 potential groups of buyers: 
1. Budget-concious
- To consider purchasing units in the North region of Singapore as it generally has the lowest prices compared to the other regions, if location is not a concern
- Likely due to towns in the North are non mature

2. Family-oriented
- To consider 4-room, 5-room and executive flat types 
- Can still cost-save by considering units the North, West and North-east regions

3. Design-concious
- To consider looking at flat models instead of flat types and locations
- Maisonette and private built flat models have bigger floor area, "luxrious feel"
- However the trade-off is that prices might be higher than the standard HDB units

However there are also other potential factors that buyers can consider, based on domain knowledge, which are not in our dataset
- Noise i.e. surrounding ambient noise of the unit
- HDB measures i.e. cooling measures, eligibility controls, ethnic quotas
- Other preferential & community features i.e. corner units, orientation of unit, proximity to community amenities

## 6. Conclusion of recommendations
1. Top considerations are regions and maturity of the estate that the HDB unit is located in
2. Followed by the unit specifications of flat type, flat model, floor area
3. Personal preferences

However, since HDB units are still under purview of public housing policy and the resale market is subject to policy changes and circumstances, buyers will have to keep these in mind.


## 7. Sources of information from further research
1. [History of HDB](https://www.hdb.gov.sg/about-us/history/town-planning)
2. [Types of flat offered by HDB](https://www.hdb.gov.sg/residential/buying-a-flat/finding-a-flat/types-of-flats)
3. [$14,000 income ceiling for families, singles to buy Plus flats on resale market: Desmond Lee](https://www.straitstimes.com/singapore/housing/14000-income-ceiling-for-families-and-singles-to-buy-plus-flats-on-resale-market-desmond-lee)
4. [Mature vs Non mature towns in Singapore](https://www.hdb.gov.sg/-/media/doc/CCG/20082023-Annexes/Annex-A1.ashx))
5. [Town areas segregated by HDB](https://www.hdb.gov.sg/about-us/history/hdb-towns-your-home)
6. [Summary of cooling measures on HDB prices](https://www.businesstimes.com.sg/property/mobile-spotlight/summary-singapores-property-cooling-measures-1996-present-day)
7. [Effect of cooling measures on HDB prices in 2013 and 2018](https://www.channelnewsasia.com/singapore/property-cooling-measures-hdb-resale-prices-2013-2018-each-singapore-town-2385831)
8. [Ethnic Quota effect on selling and buying HDB](https://www.propertyguru.com.sg/property-guides/how-the-ethnic-quota-can-affect-your-sellingbuying-ability-6747)
9. [Living with noise pollution: Serangoon, Bukit Timah and Clementi among the noisiest neighbourhoods in Singapore](https://www.straitstimes.com/singapore/housing/sounds-awful-cant-sleep-cant-talk-because-of-noise)
10. [Outside noise could affect home prices](https://cos.sg/blog-post/outside-noise-could-affect-home-prices/)
