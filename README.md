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
However, my comparison shows that some are more consistent that the others, among them **Nominatim** and GoogleV3.
ArcGis usually returns the largest bounding boxes, and AzureMaps has - literally - "far out" results on several locations.


## Following is the complete report:

<!DOCTYPE html>
<html lang="en">
<head>
<title>Geocoders Comparison Report</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
* {
    box-sizing: border-box;
}

body {
    font-family: Arial, Helvetica, sans-serif;
}

/* Style the header */
header {
    background-color: #006699;
    padding: 2px;
    text-align: center;
    font-size: 20px;
    color: white;
}

/* Create two columns/boxes that floats next to each other */
nav {
    float: left;
    width: 5%;
    background: #ccc;
    padding: 20px;
}

/* Style the list inside the menu */
nav ul {
    list-style-type: none;
    padding: 0;
}

article {
    float: left;
    margin-left: 5px;
    padding: 20px;
    width: 100%
    background-color: #f1f1f1;
}

/* centered paragraph: */
p.center {
    text-align: center;
    color: red;
}

/* all HTML elements with class="center" will be center-aligned: */
/* has no effect if the width property is not set (or set to 100%). */
.center {
    margin: auto;
    text-align: center;
    width: 50%;
    border: 3px solid green;
    padding: 10px;
}

/* Clear floats after the columns */
section:after {
    content: "";
    display: table;
    clear: both;
}

/* Create two equal columns headers */
.col_hdr {
    float: left;
    width: 50%;
    padding: 5px;
    background-color: #c2f0f0;
    text-align: center;
}
/* Create two equal columns that floats next to each other */
.column {
    float: left;
    width: 50%;
    padding: 5px;
}
/* hdr spanning a row */    
.col1_hdr {
    float: left;
    width: 100%;
    padding: 5px;
    background-color: #c2f0f0;
    text-align: center;
}
/* to accommodate small text| large pic */
.column30 {
    float: left;
    width: 35%;
    padding: 15px;
}
.column70 {
    float: left;
    width: 65%;
    padding: 5px;
}
    
/* Clear floats after the columns */
.row:after {
    content: "";
    display: table;
    clear: both;
}

iframe.pic {
margin: auto;
border: none;
height: 410px; 
width: 100%;
} 

/* style for an element with id="hide_txt" */
#hide_txt {
    color: #c2f0f0;
}

/* Style the footer */
footer {
    background-color: #006699;
    padding: 10px;
    text-align: center;
    color: white;
}

/* Responsive layout - makes the two columns/boxes stack on top of each other instead of next to each other, on small screens */
@media (max-width: 600px) {
    nav, article {
        width: 100%;
        height: auto;
    }
}
<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">\n

</style>

</head>
<body>

<header>
  <h2>Comparison of four geocoders results for a set of queries</h2>
  <h3>...You don't always get what you asked for.</h3>
</header>


<section>  

   <div width=100%;  text-align: center;>
        <h2>The geocoders being compared are: Nominatim, GoogleV3, ArcGis and AzureMaps.</h2>
         <p> I used Geopy to access the geocoders except for AzureMaps, which was functional until September 12, 2018 when
                 the same queries resulted in a "bad HTTP request" (as yet unresolved). 
                 I reverted to using the request library to for this geocoder. </p>
        
   </div>

    <div width=100%> 
        <article>
            <h2>Overview</h2>
            <p>Overall, the results differ more in the bounding boxes than in the location coordinates (the Location matrix shows less red). 
                  Also, when one geocoder has a "far out" results, as is the case for AzureMaps for 'Kings county' (Brooklyn) and 'Richmond 
                  county' (Staten Island), the same goes with the box corners distance difference. Moreover, the greater difference in the 
                  bounding boxes is often on the Eastern edge, as seen in the maps of Staten Island (panel H) or Boston (panel I) below.
            </p>
            <div class="row">
                <div class="column30">
                    <p></p>
                    <p></p>
                     <p></p>
                    <p></p>
                    <h3>Quantitative sumary: pairwise differences as distances</h3>
                    <p></p>
                    <p>The highlights make it easy to see where the geocoders differ:
                          the redder the color, the greater the difference.</p>
                    <p>Curiously, the box coordinates often agree on one side, but not the other. For Boston, the Western edge is
                        similarly located, but the Eastern one can differ by about 15 miles. </p>
                </div>
                <div class="column70">
                    <img src="./GeocodersComparison/images/Heatmap_sns_geodist_difference_mi.svg" 
                         style="width:940px; height:420px"
                         title="Pairwise differences in miles" 
                         alt="Pairwise differences in miles">
                    <!--                                     style="width:920px;"/-->
               </div>
            </div>
            
            <div class="row">
                <div class="column">
                    <h2>So where do those bounding boxes come from?</h2>
                    <p>Being a novice in GIS, I naively thought that the bounding boxes came from the shapefile bounds; 
                        the box being the smallest regular polygon covering the shape extent. To verify my assumption, I 
                        downloaded the shapefiles (including the maritime area) for the two cities in my query list: New York City and Boston. 
                        Well, it seems that the shapefiles ARE used to define the bounding boxes, yet with a lot of inconsistency:
                    </p>
                    <p></p>
                    <p>Among the four geocoders, ArcGis is the only one using the shapefiles WITHOUT the water area, as suggested by the 
                        results for Brooklyn (panel F), which is ArcGis' single correct bounding box as this geocoders returns bounding 
                        box coordinates that usually 'overshoot' the shapefile area, most often Northward.
                    </p>
                    <p>There is another type of disagreement besides the (non)covering of the areas by the bounding boxes: three out of four
                        geocoders make no distinction between New York City from New York county (Manhattan), only Nominatim does.
                    </p>
                </div>
                <div class="column">
                    <img src="./GeocodersComparison/images/comp_NYC_NYcnty_tbl.svg" alt="Table 2" />
                </div>
            </div>
            <div class="row">
                <div class="column">
                    <h2>But wait, there's more!</h2>
                    <p>ArcGis always returns the location coordinates as the center of the bounding box, while AzureMaps never does. 
                            GoogleV3 and Nominatim do so for one particular location, which is actually a monument, but Nominatim also returns 
                            location as the box center for a single county, showing some inconsistency, albeit its results are similar to Google's.
                    </p>
                </div>
                <div class="column">
                    <img src="./GeocodersComparison/images/comp_Loc_center_tbl.svg" alt="Table 1" />
                    <!-- width="240" height="180" border="10" /-->
                </div>
            </div>
        
        </article>
    </div>
        
    <div class="row">
        <div class="col_hdr" >
            <h3>A. New York City</h3>
        </div>
        <div class="col_hdr">
            <h3>B. New York county</h3>
        </div>
    </div>
    <div class="row">
        <div class="column">
            <!--- NYC !--->
           <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/New_York_City.html" ></iframe>
        </div>
        <div class="column">
           <!--- NY county !--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/New_York_county.html" ></iframe>
        </div>
     </div>
     
       
    <div class="row">
       <div class="col_hdr" >
            <h3>C. Cleopatra's Needle in Central Park</h3>
        </div>
        <div class="col_hdr" >
            <h3>D. Same, zoomed</h3>
        </div>
    </div>
    <div class="row">
        <div class="column">
            <!--- Cleo 1!--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Cleopatra's_needle.html" ></iframe>
        </div>
        <div class="column">
            <!--- Cleo 2!--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Cleopatra's_needle_zoomed.html" ></iframe>
        </div>
    </div>
        
    <div class="row">
       <div class="col_hdr">
            <h3>E. Bronx</h3>
        </div>
        <div class="col_hdr" >
            <h3>F. Brooklyn</h3>
        </div>
    </div>
   <div class="row">
       <div class="column">
            <!--- Bronx !--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Bronx_county.html" ></iframe>
         </div>               
        <div class="column">
            <!--- Brooklyn !--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Kings_county.html" ></iframe>
        </div>
    </div>
        
    <div class="row">
       <div class="col_hdr" >
            <h3>G. Queens</h3>
        </div>
        <div class="col_hdr" >
            <h3>H. Staten Island</h3>
        </div>
    </div>
    <div class="row">
        <div class="column">
            <!---Queens !--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Queens_county.html" ></iframe>
        </div>
        <div class="column">
            <!--- Richmond !--->
            <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Richmond_county.html" ></iframe>
        </div>
    </div>
     
    <div class="row">
        <div class="col1_hdr" >
            <h3>I. Boston</h3>
        </div>
        <!--- Boston!--->
        <iframe class="pic" src="./GeocodersComparison/geodata/html_frames/Boston.html" width: 98%;></iframe>
    </div>

    <article>
  
        <h1>Finally, here is something else I learned: </h1> 
        <h2>How to estimate the resolution distance using the decimal portion of a decimal degree.</h2>
        <p>Note: this is always going to be an estimate because of this simplification:</p>
               <dl>meters per degree of latitude = meters per degree of longitude</dl>
               <p>Hence, this is an "eyeball" estimate.</p>
        <p>
            * I found an <a href="https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude"
               target="_blank">excellent resource</a> on precision for geolocations.
        </p>

        <p>Excerpts from <a href="https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude"
               target="_blank">Whuber</a>:</p>
            <ul>
                <li>The first decimal place is worth up to 11.1 km: it can distinguish the position of one large city from a neighboring large city.</li>
                <li>The second decimal place is worth up to 1.1 km: it can separate one village from the next.</li>
                <li>The third decimal place is worth up to 110 m: it can identify a large agricultural field or institutional campus.</li>
                <li>The fourth decimal place is worth up to 11 m: it can identify a parcel of land. It is comparable to the typical accuracy 
                    of an uncorrected GPS unit with no interference.</li>
                <li>The fifth decimal place is worth up to 1.1 m: it distinguish trees from each other. Accuracy to this level with commercial
                    GPS units can only be achieved with differential correction.</li>
                <li>The sixth decimal place is worth up to 0.11 m: you can use this for laying out structures in detail, for designing 
                    landscapes, building roads. It should be more than good enough for tracking movements of glaciers and rivers. This can be 
                    achieved by taking painstaking measures with GPS, such as differentially corrected GPS.</li>
                <li>The seventh decimal place is worth up to 11 mm: this is good for much surveying and is near the limit of what GPS-based 
                    techniques can achieve.</li>
                <li>The eighth decimal place is worth up to 1.1 mm: this is good for charting motions of tectonic plates and movements of 
                    volcanoes. Permanent, corrected, constantly-running GPS base stations might be able to achieve this level of accuracy.</li>
                <li>The ninth decimal place is worth up to 110 microns: we are getting into the range of microscopy. For almost any 
                    conceivable application with earth positions, this is overkill and will be more precise than the accuracy of any 
                    surveying device.</li>
                <li>Ten or more decimal places indicates a computer or calculator was used and that no attention was paid to the fact that 
                    the extra decimals are useless. Be careful, because unless you are the one reading these numbers off the device, this 
                    can indicate low quality processing!</li>
            </ul>
        </p>
    </article>

</section> 
</body>
</html>



