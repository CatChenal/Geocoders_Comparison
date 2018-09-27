Purpose
=================================

Why I seup this comparison:

In another application, I was using the New York City boroughs bounding boxes to impute missing borough names for records with geolocation information; 
After a first pass, some entries were still not imputed despite clearly belonging to a particular borough. 
I then tested several geocoders to find out if the problem was a resolution issue... 
In a sense, it is: I discovered that the geolocation coordinates for the same query - could be quite different both for point locations and their bounding boxes.

