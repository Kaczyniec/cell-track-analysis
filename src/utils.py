import os
import pandas as pd
from bs4 import BeautifulSoup

def extract_wells_info(path: str) -> pd.DataFrame:
    # Load the HTML content
    with open(os.path.join(path, "WellsInfo.html"), "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    # Find the table
    table = soup.find("table", {"class": "dataframe"})
    rows = table.find("tbody").find_all("tr")
    names = table.find("thead").find("tr")
    
    data = []
    current_row_header = None
    
    for row in rows:
        cols = row.find_all(["th", "td"])
        
        # Check if first column is a th with rowspan (new row header)
        if cols and cols[0].name == "th" and cols[0].has_attr("rowspan"):
            # This is a new row header with rowspan
            current_row_header = cols[0].text.strip()
            col_vals = [current_row_header] + [col.text.strip() for col in cols[1:]]
        elif cols and cols[0].name == "th" and not cols[0].has_attr("rowspan"):
            # This is a continuation row (rowspan from previous row applies)
            col_vals = [current_row_header] + [col.text.strip() for col in cols]
        else:
            # Handle other cases (shouldn't happen with your data structure)
            col_vals = [col.text.strip() for col in cols]
        
        data.append(col_vals)
    
    # Define column names manually since header is complex
    df = pd.DataFrame(data, columns=[name.text.strip() for name in names.find_all("th")])
    df["experiment"] = path
    
    return df.rename({'WellId': 'well'}, axis=1)