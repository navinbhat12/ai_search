import pandas as pd
import duckdb

def load_structured():
    customers = pd.read_csv("data/customers.csv")
    orders = pd.read_csv("data/orders.csv")

    duckdb.sql("CREATE TABLE customers AS SELECT * FROM customers")
    duckdb.sql("CREATE TABLE orders AS SELECT * FROM orders")

if __name__ == "__main__":
    load_structured()