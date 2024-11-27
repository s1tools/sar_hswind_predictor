
Project context
=====================

Wind waves are surface waves that derive their energy and their geophysical properties from wind 
blowing over the sea surface. However, the local wind forcing itself is not sufficient to define the local 
wave properties. While the swell component can be rather well imaged and then estimated by SAR 
instrument thanks to its quasi-linear properties, the wind-sea part presents more complex non-linearities 
and are only partially or totally removed from the SAR spectral signature. SAR-derived measurements are 
therefore difficult to use given the partial and complex spectral coverage of the wind waves. Nevertheless, 
some initiatives have shown the possibility to derive a more exhaustive description of the ocean state, 
estimating the significant wave height, comparable to altimeters ```[1], [2], [3]```. Yet, wind-sea component 
impacts the SAR spectral measurements and some SAR-derived parameters such as the sea surface 
wind, the azimuth cut-off and the wave age.
Usually, with SAR observations, geophysical features retrieval applications are developed to make 
specific decisions based on predefined rules derived from human experience, and on an understanding 
of the phenomena that modulate these observations. However, while this mastery of SAR observation 
improves with time and feedback, the definition of theoretical models to accurately process the 
geophysical variables derived from SAR would require massive investment and is becoming increasingly 
complicate.

After raising and confirming this finding, CLS as member of Sentinel-1 MPC was developed a new method to derive the wind sea
Hs from L2 SAR features using a deep-learning (DL) approach. One of the advantages of such approach 
is that the classical statistical approaches require a lot of computational power on learning a very complex 
model while machine learning algorithms are more data intensive and only in the learning phase; their 
serving service is so fast. We note also that their results are too static to cope with dynamically changing 
service environments. Moreover, the greatest added value of machine learning algorithms is their ability 
to improve over time: Machine learning technology typically improves efficiency and accuracy through 
ever-increasing amounts of processed data.
The proposed method has been tested on data from Sentinel-1A Wave Mode (WV) acquisition for 
both incidences angle 23.8° (WV1) and 36.8° (WV2) collocated with global numerical wave spectra given 
by WAVEWATCH III (WW3) over the period from July 2021 to August 2022. We used a Deep Neural Networks 
(DNN) to relate the input geophysical waves parameters derived from SAR observations and
two targets :

```
1- Hs Wind Sea derived from delineated wind sea spectrum from W3 wave spectrum
2- Phs0 : the Hs Wind Sea partition given by WW3
```

[1] Li, X. M., S. Lehner, and M. X. He. “Ocean Wave Measurements Based on Satellite Synthetic Aperture 
Radar (SAR) and Numerical Wave Model (WAM) Data – Extreme Sea State and Cross Sea Analysis.” 
International Journal of Remote Sensing 29, no. 21 (2008): 6403–16. 
https://doi.org/10.1080/01431160802175546.

[2] Pleskachevsky, Andrey, Sven Jacobsen, Björn Tings, and Egbert Schwarz. “Estimation of Sea State 
from Sentinel-1 Synthetic Aperture Radar Imagery for Maritime Situation Awareness.” International 
Journal of Remote Sensing 40, no. 11 (June 3, 2019): 4104–42.
https://doi.org/10.1080/01431161.2018.1558377.

[3] Quach, Brandon, Yannik Glaser, Justin Edward Stopa, Alexis Aurélien Mouche, and Peter Sadowski. 
“Deep Learning for Predicting Significant Wave Height From Synthetic Aperture Radar.” IEEE 
Transactions on Geoscience and Remote Sensing, 2020.
doi: 10.1109/TGRS.2020.3003839

Install of s1_hswind_predictor
==================

```bash
mkvirtualenv -p python3.9 $name_of_your_virtualenv
git clone https://gitlab.brest.cls.fr/abenchaabane/sar_hswind_predictor.git
git checkout develop
pip install -U setuptools
pip install -e s1_hswind_predictor
```

Test installation
===================
```bash
cd s1_hswind_predictor/test
bin/python3.9 test_prediction.py
```

launch inference
=================

1- Refer to ```inference/cmdline.py``` to have a clear view of the needed arguments

2- The algorithm request a file text that list the L2 OCN OSW products (netcdf files) to be inferred (required)

3- The algorithm needed AUX_ML2 to load the models and the normalizers (can be defined in the config file or in the cmdline). The AUX_ML2 can be founded in : https://sar-mpc.eu/

4- some other accessory arguments could be filled to define the format in which the results will be saved, ...

/!\ the default configuration is available in config ```/inference/config.py```

```bash
python run_inference.py --list_l2_nc path/to/file.txt
```
