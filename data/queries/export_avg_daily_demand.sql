SELECT
    a.ClientID,
    a.Lat,
    a.Lon,
    CAST(a.Kg as INTEGER) as 'Kg',
    a.ProductType
from (
    select
        ClientID,
        Lat,
        Lon,
        round(sum(Kg)/180) as 'Kg', -- quick & dirty daily avg, 180 working days in 9 months (2023)
        ProductType
    from sales
    where Year = 2023
    group by ClientID, ProductType
) a

where a.Lat BETWEEN 4.3333 AND 4.9167
AND a.Lon BETWEEN -74.3500 AND -73.9167 -- Bogota Metro Area bounding box
and a.Kg >= 0; -- Has demand
