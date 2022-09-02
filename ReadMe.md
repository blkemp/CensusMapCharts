Note: you need to add a directory for the shapefiles for the different SA regions, named "Shapefiles". These can be sourced from: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files

I'll get around to creating a proper readme soon.

Another key note, this requires the Geopandas module. For some reason this REFUSED to install on my machine through pip or conda. I had to follow the instructions from the link below to install pipwin first, then a couple of the dependencies through pipwin, then finally pip install geopandas: 
https://stackoverflow.com/questions/54734667/error-installing-geopandas-a-gdal-api-version-must-be-specified-in-anaconda 

Good to know nothing has changed working in Python over the last 10 years.
