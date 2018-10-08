# Comparison of Four Geocoders: Nominatim, GoogleV3, ArcGis and AzureMAps

## Geocoding services (via Geopy):

Obtaining the geolocation coordinates of a location specified by query string can be achieved using calls to geocoding APIs directly in a browser address box, or with
a wrapping library such as [geopy](https://geopy.readthedocs.io/en/stable/).

#### Here are the links to the geocoders geopy documentation and their respective service providers:
*  [**Nominatim**](https://wiki.openstreetmap.org/wiki/Nominatim): [OpenStreetMaps](https://wiki.openstreetmap.org/wiki/Using_OpenStreetMap)
*  [GoogleV3](https://geopy.readthedocs.io/en/stable/#googlev3): [Google Map & Places API](https://developers.google.com/maps/documentation/geocoding/start)
*  [ArcGis](https://geopy.readthedocs.io/en/stable/#ArcGis): [ERSI ArcGIS API](https://developers.arcgis.com/rest/geocode/api-reference/overview-world-geocoding-service.htm)
*  [AzureMaps](https://geopy.readthedocs.io/en/stable/#azuremaps): [Microsoft Azure Maps API](https://docs.microsoft.com/en-us/azure/azure-maps/index)

(Due to an unresolved glitch, I use the ```requests``` library to access AzureMaps.)

## Why I setup this comparison:
In another application, I was using the New York City boroughs bounding boxes to impute missing borough names for records with geolocation information. 
The most expert GIS users among you would certainly predict scattershot results from such a "corner-cutting" approach, but initially I thought mine was a brilliant way to prevent over 85,000 requests... 

After a first pass, some entries were still not imputed despite clearly belonging to a particular borough. So something was wrong!
The first reason is that bounding boxes are supposed to cover a geographical area, so a lot of information is lost, hence my idea turned out quite dopey...
The other reason &mdash; granted one would really need to work with bounding boxes &mdash; is that, depending on the geolocating service, the geolocation coordinates - for the same query - could be quite different both for point locations and their bounding boxes.

Needless to say, I had to trash my <del>brilliant</del> **novice** processing shortcut; instead, I used clustering for borough name imputation.

The [**notebook** in ./GeocodersComparison/](./GeocodersComparison/Report_Items.iynb) shows how to retrieve the data and functions.

# The main conclusion from this comparison:
Depending on the geolocating service used AND the location queried, the geolocation coordinates will be WRONG. 
As I have not checked all available geocoding services - there are to date, 47 of them available via geopy - I cannot rank them, especially since none of 
them can be set as an absolute ground-truth. 
However, my comparison shows that some are more consistent that the others, among them [**Nominatim**](https://wiki.openstreetmap.org/wiki/Nominatim), the 
geocoder of [OpenStreetMaps](https://wiki.openstreetmap.org/wiki/Using_OpenStreetMap) and [GoogleV3](https://geopy.readthedocs.io/en/stable/#googlev3), the geocoder of [Google Map & Places API](https://developers.google.com/maps/documentation/geocoding/start).
[ArcGis](https://geopy.readthedocs.io/en/stable/#ArcGis), the [ERSI ArcGIS API]() usually returns the largest bounding boxes, and AzureMaps has - literally - "far out" results on several locations.


## Following is the complete report: 
[notebook viewer](https://nbviewer.jupyter.org/github/CatChenal/Geocoders_Comparison/blob/master/GeocodersComparisonReport.ipynb)
