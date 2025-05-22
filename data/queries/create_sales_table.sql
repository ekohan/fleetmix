CREATE TABLE IF NOT EXISTS sales (
    Date TEXT,
    Day INTEGER,
    Month INTEGER,
    Year INTEGER,
    YearMonth TEXT,
    TransportID TEXT,
    ClientID TEXT,
    Material TEXT,
    Description TEXT,
    Units REAL,
    Kg REAL,
    Lat REAL,
    Lon REAL,
    ProductType TEXT,
    SourceYear INTEGER
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sales_clientid ON sales(ClientID);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(Date);
CREATE INDEX IF NOT EXISTS idx_sales_yearmonth ON sales(YearMonth);
CREATE INDEX IF NOT EXISTS idx_sales_producttype ON sales(ProductType);
CREATE INDEX IF NOT EXISTS idx_sales_sourceyear ON sales(SourceYear);