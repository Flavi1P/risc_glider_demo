# risc_glider_demo
Some notebook to share calibration and cross calibration of glider data within the RISC project

## Worflow

The user has to run the notebooks in order, starting with `00_bio_optic_calibration` and then `01_gliders_matchup_quality_assessment.ipynb`.
The idea is to first make a factorycal calibration of the bio-optical sensors, make some correction on sensor specific documented issues and add a QC. Then to assess the quality of the glider data by comparing them together and with the reference data from the ship.

### Bio optic calibration
The first notebook `00_bio_optic_calibration` is used to calibrate the bio-optical sensors of the gliders. It includes steps for:
- Converting beta to particulate backscattering
- Correcting the chlorophyll fluorescence for NPQ and dark drift 
- Separating baseline and spike from fluorescence and backscatter data
- Reconstructing light measurement near the surface
- Correcting the backscatter for the effect of bubbles

### Matchup quality assessment
The second notebook `01_gliders_matchup_quality_assessment.ipynb` is used to assess the quality of the glider data by comparing them together and with the reference data from the ship. It includes steps for:
- Looking at the distance between each platform along the mission
- Looking at the number of matchups between the gliders and the ship or between each glider withing a defined maximum spatial and temporal distance
- Looking at the R2 (goodness of fit) depending on distance for each variable between each platform
- Looking at the value of the regression

### Gliders intercalibration
This should rather be named gliders alignment. 
The main objective is to bring all gliders to a common reference, a glider that will be chosen based on the diagnostic of the previous notebook.
It will includes mainly the main steps as previously for each platform compared to the reference glider and then apply a linear regression to align the data of the gliders to the reference glider.

### Glider ship intercalibration
Here we make sure that the glider data are aligned with the ship data.
This will be done by comparing the glider data to the ship data and applying a linear regression to align the glider data to the ship data.

## General information
This repository is stil in development. Most of the functions are shared between the notebooks and hence should be in the utils folder. Some documentation is still missing. 