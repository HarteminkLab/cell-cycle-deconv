import pandas as pd
import re

def simple_df_to_latex_table(df, 
                             math_columns=None,
                             column_alignment='clc',
                             table_width='\\textwidth',
                             add_borders=True,
                             decimal_places=4,
                             include_header=True,
                             caption=None,
                             label=None):
    """
    Convert a pandas DataFrame directly to a LaTeX table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to convert
    math_columns : list or None
        List of column names that should be wrapped in LaTeX math mode ($...$)
    column_alignment : str or None
        Column alignment string (e.g., 'cccc'). If None, uses 'c' for each column
    table_width : str
        Table width for tabularx environment ('textwidth', 'linewidth', etc.)
    add_borders : bool
        Whether to include \cline borders around the table
    decimal_places : int or None
        Number of decimal places for numerical columns
    include_header : bool
        Whether to include column headers in the table
    caption : str or None
        Table caption
    label : str or None
        Table label for referencing
        
    Returns:
    --------
    str : LaTeX table string
    """
    
    def format_value(value, col_name, is_math=False):
        """Format a cell value with appropriate LaTeX formatting."""
        if pd.isna(value):
            return ""
        
        # Handle numerical formatting
        if isinstance(value, (int, float)) and decimal_places is not None:
            value = f"{value:.{decimal_places}f}"
        
        value_str = str(value)
        
        return value_str
    
    # Set up column alignment
    n_cols = len(df.columns)
    if column_alignment is None:
        column_alignment = 'c' * n_cols
    
    # Set up math columns
    if math_columns is None:
        math_columns = []
    
    # Start building LaTeX string
    latex_str = r"\setlength{\tabcolsep}{0.5em} % for the horizontal padding" + "\n"
    latex_str += r"{\renewcommand{\arraystretch}{1.2}% for the vertical padding" + "\n\n"
    
    # Table environment
    latex_str += r"\begin{table}[]" + "\n"
    
    if caption:
        latex_str += f"\\caption{{{caption}}}\n"
    if label:
        latex_str += f"\\label{{{label}}}\n"
    
    # Choose table environment based on width specification
    if table_width:
        latex_str += f"\\begin{{tabularx}}{{{table_width}}}{{{column_alignment}}}\n"
    else:
        latex_str += f"\\begin{{tabular}}{{{column_alignment}}}\n"
    
    # Top border
    if add_borders:
        latex_str += f"\\cline{{1-{n_cols}}}\n"
    
    # Header row
    if include_header:
        header_row = []
        for col in df.columns:
            is_math = col in math_columns
            formatted_col = format_value(col, col, is_math)
            if not is_math:
                formatted_col = f"\\textbf{{{formatted_col}}}"
            header_row.append(formatted_col)
        
        latex_str += " & ".join(header_row) + " \\\\\n"
        
        if add_borders:
            latex_str += f"\\cline{{1-{n_cols}}}\n"
    
    # Data rows
    for _, row in df.iterrows():
        row_cells = []
        for col in df.columns:
            is_math = col in math_columns
            formatted_value = format_value(row[col], col, is_math)
            row_cells.append(formatted_value)
        
        latex_str += " & ".join(row_cells) + " \\\\\n"
    
    # Bottom border
    if add_borders:
        latex_str += f"\\cline{{1-{n_cols}}}\n"
    
    # Close table
    if table_width:
        latex_str += "\\end{tabularx}\n"
    else:
        latex_str += "\\end{tabular}\n"
    
    latex_str += "\\end{table}\n"
    latex_str += "}"
    
    return latex_str
