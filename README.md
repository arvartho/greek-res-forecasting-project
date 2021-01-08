# Greek RES Forecasting

An attemp to generate short-term forecasts for wind and solar energy production for the Greek energy grid. Namely, the following hypothesis is explored:
> The information from weather variables of a subset of RES installation in Greece, along with the last day's production, is sufficient for accurately forecasting the next day's RES generation.

## Premise
In detail, the idea behind this exercise is to explore the predictability of the solar energy production for the Greek energy grid by using only aggregated weather information from the top 20% of wind and solar installations. The forecast horizon for this project is one week, and the predictions are performed by applying a rolling forecast with one day window and using as features the next day's weather forecast and the previous day energy production. Additionally, a number of temporal features are used as well, in an attemp to describe the seasonality of the timeseries. Features such as the month, hour, day, day of week and day of year are encoded with polar coordinates to capture their cyclical nature.


## Data
For this endeavour, a dataset for Greek RES was created using a number of different sources, by combining energy generation, geolocation and weather forecast data. The dataset was formed by locating, using the geolocation data, a subset of the installations with the greater solar and wind energy production capacity. Afterwards, the historical hourly weather forecasts of a number of weather variables for the locations of interest is extracted using the [StormGlass Weather API](https://docs.stormglass.io/#/weather?id=point-request). Whereas the energy data were collected by the [European Network of Transmission System Operators for Electricity (ENTSO-E)](https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show) and consist of the actual aggregated net solar and wind generation output in hourly intervals. For each type of energy the weather information is aggregated and combined with the output energy information.

The raw data were collected from the following sources:

* RES energy production from [ENTSO-E](https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show)
* Weather data captured using the [Stormglass API](https://docs.stormglass.io/#/weather?id=point-request)
* Wind turbine and PV installations geolocation data, for the locations that aquired operation licence as found in [Regulatory Authority for Energy (RAE)](https://geo.rae.gr/)

More information regarding energy licencing http://www.opengov.gr/minenv/?p=1031

## Forecast Modeling
The study focuses on linear and non-linear statistical models and some hybrid combination of those
* Linear Regression
* Ridge Regression
* Elastic Net Regression
* Random Forest Regression
* Extreme Gradient Boosting
* Facebook Prophet
* Hybrid Facebook Prophet with Random Forest Regression
* Hybrid Facebook Prophet with Extreme Gradient Boosting

## Experiments and evaluation
For the experiments, 4 years of weather and energy data were collected from 2017 to 2020. Namely, 5 different training and testing datasets have been generated, for each energy source. The testing is executed on specific weeks of **March, May, July, October and December of 2020**. The forecast evaluation is perfoemed using MAE and RMSE as well as the violin plots of the error distribution for each week.

## Conclusion
The experiments draw the conclusion that the non-linear models (Random Forest Regression and Extreme Gradient Boost) can be better predictors of the renewble energy in the Greek grid. Especially when it comes to wind energy forecasts, the RF and XGB, perform much better than the linear and hybrid models. This is consistent with the bibliography that suggests that the volatile nature of wind can better captured by non-linear models.
When it comes to solar energy, which is less volatile and with more predictable seasonality, the performance of linear and non-linear models is comperable, although RF and XGB still outperform their counterparts. 
Additionally, the experiments that were conducted in different weekly time frames across the year, indicate that soral energy output during summer and winter (July and January) months is more predictable than the rest of the year. On the other hand wind energy generation has no obvious seasonal pattern, thus can be equally volatile and challenging to predict throughout the year.

---

## Future Work
* Improve tuning of FB Prophet model
* Feature selection with Maximum relevance minimum redundancy (MRMR) (pip install Pymrmre)
* Attemp forecasts using as features the weather variables, the actual energy data up until two days before the target day (t-2 actual lagged values) and the energy forecast of the day before (t-1 lagged energy forecast)
* Experiment with wavelet features and wavelet decomposition of the timeseries
* Using exhaustive parameter tuning techniques such as grid search can further improve models' prediction performance

## References
[1] D. A. Wood, “Hourly-averaged solar plus wind power generation for Germany 2016: Long-term prediction, short-term forecasting, data mining and outlier analysis,” Sustain. Cities Soc., vol. 60, no. April, p. 102227, 2020, doi: 10.1016/j.scs.2020.102227.

[2] M. Bouzerdoum, A. Mellit, and A. Massi Pavan, “A hybrid model (SARIMA-SVM) for short-term power forecasting of a small-scale grid-connected photovoltaic plant,” Sol. Energy, vol. 98, no. PC, pp. 226–235, 2013, doi: 10.1016/j.solener.2013.10.002.

[3] N. Bigdeli, M. Salehi Borujeni, and K. Afshar, “Time series analysis and short-term forecasting of solar irradiation, a new hybrid approach,” Swarm Evol. Comput., vol. 34, pp. 75–88, 2017, doi: 10.1016/j.swevo.2016.12.004.

[4] C. Voyant, C. Paoli, M. Muselli, and M. L. Nivet, “Multi-horizon solar radiation forecasting for Mediterranean locations using time series models,” Renew. Sustain. Energy Rev., vol. 28, pp. 44–52, 2013, doi: 10.1016/j.rser.2013.07.058.