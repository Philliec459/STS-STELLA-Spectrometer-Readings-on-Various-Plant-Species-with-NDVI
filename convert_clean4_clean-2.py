#!/usr/bin/env python
# coding: utf-8

# # **Strip STELLA-Q2 data for Timestamp and Channel irradiance data**
# ---
# 
# ## Conversion Code to write data and Plots to Excel
# ---
# 
# 
# ## Read in the **data.csv** type file that has been converted to an Excel file and the White-Card Excel file too:

# In[1]:


import pandas as pd
import xlsxwriter


# Relative Path for STELLA Raw and White-Card Excel files
data_csv_xlsx_path    = r'./data_Nick_HomeDepot.xlsx'
white_card_xlsx_path  = r'./data_Nick_HomeDepot_white.xlsx'


# ## 1) Write Raw and White Card Irradiance data to **1_filtered_data.xlsx** with easy to read columns:
# ---

# In[2]:


# Read the Excel file
df1 = pd.read_excel(data_csv_xlsx_path,index_col=False)
df2 = pd.read_excel(white_card_xlsx_path,index_col=False, nrows = 1)
#df1 = pd.read_excel('data_Nick_HomeDepot.xlsx')
#df2 = pd.read_excel('data_Nick_HomeDepot_white.xlsx')


# Remove leading/trailing whitespaces in column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Find the column containing 'timestamp' in its name
timestamp_column_1 = [col for col in df1.columns if 'timestamp' in col.lower()][0]
timestamp_column_2 = [col for col in df2.columns if 'timestamp' in col.lower()][0]

# Extract wavelength values from column names and add 'nm'
new_column_headings_1 = [col.split('_')[1] + 'nm' if 'nm' not in col.split('_')[1] else col.split('_')[1] for col in df1.columns if 'irradiance_' in col and '_irradiance_uW_per_cm_squared' in col]
new_column_headings_2 = [col.split('_')[1] + 'nm' if 'nm' not in col.split('_')[1] else col.split('_')[1] for col in df2.columns if 'irradiance_' in col and '_irradiance_uW_per_cm_squared' in col]

# Filter the columns based on your criteria
filtered_columns_1 = ['Test', 'batch', timestamp_column_1] + [col for col in df1.columns if 'irradiance_' in col and '_irradiance_uW_per_cm_squared' in col]
filtered_columns_2 = ['Test', 'batch', timestamp_column_2] + [col for col in df2.columns if 'irradiance_' in col and '_irradiance_uW_per_cm_squared' in col]

# Subset the DataFrame with the filtered columns
filtered_data_1 = df1[filtered_columns_1]
filtered_data_2 = df2[filtered_columns_2]

# Rename the columns
filtered_data_1.columns = ['Test', 'batch', timestamp_column_1] + new_column_headings_1
filtered_data_2.columns = ['Test', 'batch', timestamp_column_2] + new_column_headings_2

# Write the filtered data to a new Excel file
with pd.ExcelWriter('1_filtered_data.xlsx') as writer:
    # Write the filtered data to the 'Raw' tab
    df1.to_excel(writer, sheet_name='Original Data', index=False)

    # Write the filtered data to the 'Raw' tab
    filtered_data_1.to_excel(writer, sheet_name='Raw Filtered', index=False)

    # Write the filtered data to the 'White' tab
    filtered_data_2.to_excel(writer, sheet_name='White Card Filtered', index=False)


# ## 2) Write Raw, White Card Irradiance and White Card Corrected data to **2_white_card_corrected.xlsx**:
# ---

# In[3]:


# Read the Excel files
df_raw = pd.read_excel('1_filtered_data.xlsx', sheet_name='Raw Filtered')
df_white = pd.read_excel('1_filtered_data.xlsx', sheet_name='White Card Filtered')

# Divide each irradiance channel in the Raw tab by the first irradiance value in the White Card tab
df_raw_corrected = df_raw.copy()
for col in df_raw.columns:
    if 'nm' in col:
        wavelength = col.split('nm')[0]
        df_raw_corrected[col] = df_raw[col] / df_white[wavelength + 'nm'].iloc[0]

# Create a new Excel file
with pd.ExcelWriter('2_white_card_corrected.xlsx', engine='xlsxwriter') as writer:
    # Write the filtered data to the 'Raw' tab
    df1.to_excel(writer, sheet_name='Original Data', index=False)
    
    # Add the Raw data
    df_raw.to_excel(writer, sheet_name='Raw Filtered', index=False)
    
    # Add the White Card data
    df_white.to_excel(writer, sheet_name='White Card Filtered', index=False)
    
    # Add the corrected Raw data
    df_raw_corrected.to_excel(writer, sheet_name='White Card Corrected', index=False)


# ## 3) Write Raw and White Card Irradiance data from filtered_data.xlsx to create Time-Series plots for **3_Raw_filtered_data_with_aggregated_line_time_series_plots.xlsx** Excel File:
# ---

# In[4]:


# Read data from both tabs
df_raw = pd.read_excel('1_filtered_data.xlsx', sheet_name='Raw Filtered')
df_white = pd.read_excel('1_filtered_data.xlsx', sheet_name='White Card Filtered')

# Create a new Excel file
with pd.ExcelWriter('3_Raw_filtered_data_with_aggregated_line_time_series_plots.xlsx', engine='xlsxwriter') as writer:
    # Write the Raw data to the Excel file
    df_raw.to_excel(writer, sheet_name='Raw', index=False)
    
    # Write the White Card data to the Excel file
    df_white.to_excel(writer, sheet_name='White Card', index=False)
    
    # Add line plot for Raw data
    workbook = writer.book
    worksheet_raw = writer.sheets['Raw']
    worksheet_white = writer.sheets['White Card']
    
    # Aggregate data for Raw tab
    timestamps_raw = df_raw['timestamp_iso8601']
    wavelengths_raw = [col for col in df_raw.columns if col.endswith('nm')]
    values_raw = df_raw[wavelengths_raw].astype(float)
    
    # Add line plot for Raw data
    chart_raw = workbook.add_chart({'type': 'line'})
    for col_idx, col_name in enumerate(wavelengths_raw):
        chart_raw.add_series({
            'name': f'{col_name}',
            'categories': ['Raw', 1, 0, len(timestamps_raw), 0],
            'values': ['Raw', 1, col_idx + 1, len(timestamps_raw), col_idx + 1],
        })
    chart_raw.set_title({'name': 'Raw Time-Series Data'})
    chart_raw.set_x_axis({'name': 'Timestamp'})
    chart_raw.set_y_axis({'name': 'Wavelength'})
    worksheet_raw.insert_chart('E2', chart_raw)
    
    # Aggregate data for White Card tab
    timestamps_white = df_white['timestamp_iso8601']
    wavelengths_white = [col for col in df_white.columns if col.endswith('nm')]
    values_white = df_white[wavelengths_white].astype(float)
    
    # Add line plot for White Card data
    chart_white = workbook.add_chart({'type': 'line'})
    for col_idx, col_name in enumerate(wavelengths_white):
        chart_white.add_series({
            'name': f'{col_name}',
            'categories': ['White Card', 1, 0, len(timestamps_white), 0],
            'values': ['White Card', 1, col_idx + 1, len(timestamps_white), col_idx + 1],
        })
    chart_white.set_title({'name': 'White Card Time-Series Data'})
    chart_white.set_x_axis({'name': 'Timestamp'})
    chart_white.set_y_axis({'name': 'Wavelength', 'min': min(values_white.min()), 'max': max(values_white.max())})
    worksheet_white.insert_chart('E2', chart_white)


# ## 4) Wavelength plots for Raw and White Card data written to **4_Raw_wavelength_plots.xlsx** Excel File:
# ---

# In[5]:


# Read the Excel files
df_raw = pd.read_excel('1_filtered_data.xlsx', sheet_name='Raw Filtered')
df_white = pd.read_excel('1_filtered_data.xlsx', sheet_name='White Card Filtered')

# Create a new Excel file
with pd.ExcelWriter('4_Raw_wavelength_plots.xlsx', engine='xlsxwriter') as writer:
    # Add the Raw data
    df_raw.to_excel(writer, sheet_name='Raw', index=False)
    workbook = writer.book
    worksheet_raw = writer.sheets['Raw']

    # Add a new worksheet for the Raw data plots
    chart_worksheet_raw = workbook.add_worksheet('Raw Plots')

    # Extract wavelengths and time stamps for Raw data
    wavelengths_raw = [col.split('nm')[0] for col in df_raw.columns if 'nm' in col]
    timestamps_raw = df_raw['timestamp_iso8601']

    # Write the wavelengths to the worksheet
    chart_worksheet_raw.write_row('A1', ['Timestamp'] + wavelengths_raw)

    # Write the timestamps and irradiance values to the worksheet
    for i, ts in enumerate(timestamps_raw, start=1):
        chart_worksheet_raw.write(i, 0, ts)
        chart_worksheet_raw.write_row(i, 1, df_raw.iloc[i - 1][[col + 'nm' for col in wavelengths_raw]])

    # Create a line chart for all rows in Raw data
    chart_raw = workbook.add_chart({'type': 'line'})
    for i in range(len(timestamps_raw)):
        chart_raw.add_series({
            'name': f'Row {i + 1}',
            'categories': ['Raw Plots', 1, 0, len(timestamps_raw), 0],
            'values': ['Raw Plots', 1 + i, 1, 1 + i, len(wavelengths_raw)],
        })
    chart_raw.set_x_axis({'name': 'Timestamp'})
    chart_raw.set_y_axis({'name': 'Irradiance'})
    chart_raw.set_title({'name': 'Raw Data Wavelenght Plot'})
    chart_worksheet_raw.insert_chart('E1', chart_raw)

    
    
    
    
    # Add the White Card data
    df_white.to_excel(writer, sheet_name='White Card', index=False)
    worksheet_white = writer.sheets['White Card']

    # Add a new worksheet for the White Card data plots
    chart_worksheet_white = workbook.add_worksheet('White Card Plots')

    # Extract wavelengths and time stamps for White Card data
    wavelengths_white = [col.split('nm')[0] for col in df_white.columns if 'nm' in col]
    timestamps_white = df_white['timestamp_iso8601']

    # Write the wavelengths to the worksheet
    chart_worksheet_white.write_row('A1', ['Timestamp'] + wavelengths_white)

    # Write the timestamps and irradiance values to the worksheet
    for i, ts in enumerate(timestamps_white, start=1):
        chart_worksheet_white.write(i, 0, ts)
        chart_worksheet_white.write_row(i, 1, df_white.iloc[i - 1][[col + 'nm' for col in wavelengths_white]])

    # Create a line chart for all rows in White Card data
    chart_white = workbook.add_chart({'type': 'line'})
    for i in range(len(timestamps_white)):
        chart_white.add_series({
            'name': f'Row {i + 1}',
            'categories': ['White Card Plots', 1, 0, len(timestamps_white), 0],
            'values': ['White Card Plots', 1 + i, 1, 1 + i, len(wavelengths_white)],
        })
    chart_white.set_x_axis({'name': 'Timestamp'})
    chart_white.set_y_axis({'name': 'Irradiance'})
    chart_white.set_title({'name': 'Raw White Card Data Wavelength Plot'})
    chart_worksheet_white.insert_chart('E1', chart_white)


# ## 5) Wavelength plots for Raw and White Card Corrected Data written to **5_wavelength_plots_White_Card_Corr_NDVI.xlsx** Excel File with NDVI time-series plot with Test as x-axis:
# ---

# In[6]:


# Read the Excel files
df_raw = pd.read_excel('2_white_card_corrected.xlsx', sheet_name='Raw Filtered')
df_white = pd.read_excel('2_white_card_corrected.xlsx', sheet_name='White Card Corrected')
#df_white_corr = pd.read_excel('2_white_card_corrected.xlsx', sheet_name='White Card Corrected')

# Calculate NDVI
#ndvi_values = (near IR          -       Red        ) / (near IR                  + Red       )
ndvi_values = (df_white['860nm'] - df_white['645nm']) / (df_white['860nm'] + df_white['645nm'])

# Create a new Excel file
with pd.ExcelWriter('5_wavelength_plots_White_Card_Corrected_NDVI.xlsx', engine='xlsxwriter') as writer:


    
    
    
    # Add the Raw data
    df_raw.to_excel(writer, sheet_name='Raw', index=False)
    workbook = writer.book
    worksheet_raw = writer.sheets['Raw']

    # Add a new worksheet for the Raw data plots
    chart_worksheet_raw = workbook.add_worksheet('Raw Plots')

    # Extract wavelengths and time stamps for Raw data
    wavelengths_raw = [col.split('nm')[0] for col in df_raw.columns if 'nm' in col]
    timestamps_raw = df_raw['timestamp_iso8601']

    # Write the wavelengths to the worksheet
    chart_worksheet_raw.write_row('A1', ['Timestamp'] + wavelengths_raw)

    # Write the timestamps and irradiance values to the worksheet
    for i, ts in enumerate(timestamps_raw, start=1):
        chart_worksheet_raw.write(i, 0, ts)
        chart_worksheet_raw.write_row(i, 1, df_raw.iloc[i - 1][[col + 'nm' for col in wavelengths_raw]])

    # Create a line chart for all rows in Raw data
    chart_raw = workbook.add_chart({'type': 'line'})
    for i in range(len(timestamps_raw)):
        chart_raw.add_series({
            'name': f'Row {i + 1}',
            'categories': ['Raw Plots', 1, 0, len(timestamps_raw), 0],
            'values': ['Raw Plots', 1 + i, 1, 1 + i, len(wavelengths_raw)],
        })
    chart_raw.set_x_axis({'name': 'Timestamp'})
    chart_raw.set_y_axis({'name': 'Irradiance'})
    chart_raw.set_title({'name': 'Raw Data Wavelenght Plot'})
    chart_worksheet_raw.insert_chart('E1', chart_raw)


    
    
    
    
   # Add the White Card data
    df_white.to_excel(writer, sheet_name='White Card Corrected', index=False)
    worksheet_white = writer.sheets['White Card Corrected']

    # Add a new worksheet for the White Card data plots
    chart_worksheet_white = workbook.add_worksheet('White Card Corr Plots')

    # Extract wavelengths and time stamps for White Card data
    wavelengths_white = [col.split('nm')[0] for col in df_white.columns if 'nm' in col]
    timestamps_white = df_white['timestamp_iso8601']

    # Write the wavelengths to the worksheet
    chart_worksheet_white.write_row('A1', ['Timestamp'] + wavelengths_white)

    # Write the timestamps and irradiance values to the worksheet
    for i, ts in enumerate(timestamps_white, start=1):
        chart_worksheet_white.write(i, 0, ts)
        chart_worksheet_white.write_row(i, 1, df_white.iloc[i - 1][[col + 'nm' for col in wavelengths_white]])

    # Create a line chart for all rows in White Card data
    chart_white = workbook.add_chart({'type': 'line'})
    for i in range(len(timestamps_white)):
        chart_white.add_series({
            'name': f'Row {i + 1}',
            'categories': ['White Card Corr Plots', 1, 0, len(timestamps_white), 0],
            'values': ['White Card Corr Plots', 1 + i, 1, 1 + i, len(wavelengths_white)],
        })
    chart_white.set_x_axis({'name': 'Timestamp'})
    chart_white.set_y_axis({'name': 'Irradiance'})
    chart_white.set_title({'name': 'White Card Corrected Wavelength Plot'})
    chart_worksheet_white.insert_chart('E1', chart_white)



    
    
    
    
    
    # Add NDVI data to a new worksheet
    ndvi_sheet_name = 'NDVI'
    ndvi_df = pd.DataFrame({'Test': df_white['Test'], 'NDVI': ndvi_values})
    ndvi_df.to_excel(writer, sheet_name=ndvi_sheet_name, index=False)

    # Create a new workbook object
    workbook = writer.book
    
    # Create a new worksheet for the NDVI plot
    ndvi_chart_worksheet = workbook.add_worksheet('NDVI Plot')

    # Write the Test and NDVI values to the worksheet
    ndvi_chart_worksheet.write_row('A1', ['Test', 'NDVI'])
    ndvi_chart_worksheet.write_column('A2', ndvi_df['Test'])
    ndvi_chart_worksheet.write_column('B2', ndvi_df['NDVI'])

    # Create a line chart for NDVI
    ndvi_chart = workbook.add_chart({'type': 'line'})
    #ndvi_chart = workbook.add_chart({'type': 'line', 'size': {'width': 1800, 'height': 1600}})

    ndvi_chart.add_series({
        'categories': [ndvi_sheet_name, 1, 0, len(ndvi_values), 0],
        'values': [ndvi_sheet_name, 1, 1, len(ndvi_values), 1],
        'name': 'NDVI',
        'line': {'color': 'green', 'dash_type': 'dash'},
        'marker': {'type': 'circle', 'size': 8, 'fill': {'color': 'green'}, 'border': {'color': 'green'}}
       })
    ndvi_chart.set_x_axis({'name': 'Test'})
    
    #ndvi_chart.set_y_axis({'name': 'NDVI'})
    ndvi_chart.set_y_axis({'name': 'NDVI', 'min': 0, 'max': 1})  # Set the y-axis limits

    ndvi_chart.set_title({'name': 'NDVI Time Series Plot'})
    
    # Set the dimensions of the chart
    ndvi_chart.set_size({'width': 1200, 'height': 600})
    
    ndvi_chart_worksheet.insert_chart('C2', ndvi_chart)
    
    


# ## End of Code
# ---
# ---

# In[ ]:




