select
    a.ClientID,
    a.Lat,
    a.Lon,
    CAST(a.Kg AS INTEGER) as 'Kg',
    a.ProductType
from (
    select
        ClientID,
        Lat,
        Lon,
        round(sum(Kg)/20) as 'Kg', -- quick & dirty daily avg, 20 working days in a month (September here)
        ProductType
    from sales
    where YearMonth = '2023.09'
    group by ClientID, ProductType
) a

where a.Lat BETWEEN 4.3333 AND 4.9167
AND a.Lon BETWEEN -74.3500 AND -73.9167 -- Bogota Metro Area bounding box
and a.Kg >= 0; -- Has demand
