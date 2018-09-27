Conclusions
=================================

# The main conclusion from this comparison:

Depending on the geolocating service used AND the location queried, the geolocation coordinates will be WRONG. 
As I have not checked all available geocoding services - there are to date, 47 of them available via geopy - I cannot rank them, especially since none of 
them can be set as an absolute ground-truth. 
However, my comparison of four geocoders (Nominatim, GoogleV3, ArcGis and AzureMaps), shows some are more consistent that the others, among them Nominatim, the geocoder of OpenStreetMaps and GoogleV3, the geocoder of GooglePlaces API.
ArcGis usually returns the largest bounding boxes, and AzureMaps has - literally - "far out" results on several locations.
