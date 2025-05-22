SELECT
    ClientID,
    Lat,
    Lon,
    CAST(ROUND(Kg) AS INTEGER) as Kg,
    ProductType
FROM (
    SELECT
        ClientID,
        Lat,
        Lon,
        SUM(Kg)/180 as Kg,  -- 180 working days in 9 months
        ProductType
    FROM sales
    WHERE SourceYear = 2023
    GROUP BY ClientID, ProductType
) a
WHERE Lat BETWEEN 4.3333 AND 4.9167
AND Lon BETWEEN -74.3500 AND -73.9167  -- Bogota Metro Area bounding box
AND Kg >= 0  -- Has demand
AND CAST(ClientID AS INTEGER) % 7 = 0;  -- Select approximately 15% of clients
