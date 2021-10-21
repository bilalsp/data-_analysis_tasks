from typing import Tuple
import xlrd


def load_data(file_path: str, sheet_name: str, skip_rows: int = 0,
              rows_limit: int = None, verbose=True) -> Tuple[str, dict]:
    """Load the excel data using xlrd.
    
    Args:
        file_path: excel file path
        sheet_name: sheet name in a workbook
        skip_rows: number of header rows which should be skiped
        rows_limit: number of data rows
        verbose: 

    Returns:
        data: as a string
        desc: decription about the data
    """
    if verbose:
        print('Loading data from {}...'.format(file_path))

    with xlrd.open_workbook(file_path, on_demand=True) as book:
        sheet = book.sheet_by_name(sheet_name) # Only loads the specified sheet
    book.release_resources()

    nrows = sheet.nrows
    ncols = sheet.ncols
    last_row_idx = nrows if rows_limit is None else min(nrows, skip_rows+rows_limit+1)
    
    data = []
    for row_idx in range(skip_rows, last_row_idx):
        row = ",".join(map(str, sheet.row_values(row_idx)))
        data.append(row)
    data = "\n".join(data)

    desc = {}
    desc['column_name'] = sheet.row_values(skip_rows-1)
    desc['ncols'] = ncols

    if verbose:
        print('Data has loaded.')

    return data, desc
