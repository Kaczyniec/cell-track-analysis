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
    
    data =[]
    current_row_header = None
    
    for row in rows:
        # Extract all table headers and data cells in the row
        cols = row.find_all(["th", "td"])
        
        # Count the <th> tags to determine the row type
        th_tags = [col for col in cols if col.name == "th"]
        
        # If there are 2 or more <th> tags, it's a new row header group
        if len(th_tags) >= 2:
            current_row_header = th_tags[0].text.strip()
            # We can just extract all columns cleanly here
            col_vals =[col.text.strip() for col in cols]
            
        # If there is exactly 1 <th> tag, it's a continuation row
        elif len(th_tags) == 1:
            # Prepend the stored header from the previous rows
            col_vals = [current_row_header] +[col.text.strip() for col in cols]
            
        else:
            # Fallback (shouldn't happen with your data structure)
            col_vals = [col.text.strip() for col in cols]
        
        data.append(col_vals)
    
    # Define column names manually since header is complex
    columns =[name.text.strip() for name in names.find_all("th")]
    df = pd.DataFrame(data, columns=columns)
    df["experiment"] = path
    
    return df.rename({'WellId': 'well'}, axis=1)