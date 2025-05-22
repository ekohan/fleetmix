SELECT
    Date,
    ClientID,
    Lat,
    Lon,
    CAST(SUM(Kg) as INTEGER) as Kg,
    ProductType
FROM sales
WHERE Lat BETWEEN 4.3333 AND 4.9167 
    AND Lon BETWEEN -74.3500 AND -73.9167  -- Bogota Metro Area bounding box
    AND Date = '2024-08-31'  -- Filter for specific day
GROUP BY Date, ClientID, ProductType, Lat, Lon
HAVING SUM(Kg) >= 30  -- Remove small demand at client level
    AND SUM(Kg) <= 4000  -- No outliers at client level

