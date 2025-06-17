import numpy as np
import pandas as pd


def define_new_strand_specific_key(df, new_key, watson_key, crick_key, key_type=int):
    df = df.copy()
    sel_watson = df.strand == '+'
    sel_crick = df.strand == '-'

    df.loc[sel_watson, new_key] = df.loc[sel_watson][watson_key]
    df.loc[sel_crick, new_key] = df.loc[sel_crick][crick_key]
    df[new_key] = df[new_key].astype(key_type)

    return df


def define_genomic_region(df, origin_key, offset_span, new_keys, strand_aware=True):
    """
    Define a genomic region based on an origin coordinate and offset span
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract parameters
    start_offset, end_offset = offset_span
    start_col, end_col = new_keys
    
    # Check if origin column exists
    if origin_key not in df_copy.columns:
        raise ValueError(f"Origin column '{origin_key}' not found in dataset")
    
    # Initialize new columns with NaN
    df_copy[start_col] = np.nan
    df_copy[end_col] = np.nan
    
    # Find rows with valid origin coordinates
    mask_valid = df_copy[origin_key].notna()
    
    if strand_aware and 'strand' in df_copy.columns:
        # Strand-aware calculation
        mask_plus = mask_valid & (df_copy['strand'] == '+')
        mask_minus = mask_valid & (df_copy['strand'] == '-')
        
        # For + strand: apply offsets directly
        df_copy.loc[mask_plus, start_col] = df_copy.loc[mask_plus, origin_key] + start_offset
        df_copy.loc[mask_plus, end_col] = df_copy.loc[mask_plus, origin_key] + end_offset
        
        # For - strand: reverse the offsets (upstream is higher coordinate)
        df_copy.loc[mask_minus, start_col] = df_copy.loc[mask_minus, origin_key] - end_offset
        df_copy.loc[mask_minus, end_col] = df_copy.loc[mask_minus, origin_key] - start_offset
        
        # Ensure start <= end for all rows
        df_copy.loc[mask_minus, [start_col, end_col]] = df_copy.loc[mask_minus, [end_col, start_col]].values
        
    else:
        # Simple coordinate-based calculation (not strand-aware)
        df_copy.loc[mask_valid, start_col] = df_copy.loc[mask_valid, origin_key] + start_offset
        df_copy.loc[mask_valid, end_col] = df_copy.loc[mask_valid, origin_key] + end_offset
    
    # Convert to integers where possible (genomic coordinates are typically integers)
    df_copy[start_col] = df_copy[start_col].astype('Int64')  # Nullable integer type
    df_copy[end_col] = df_copy[end_col].astype('Int64')
    
    return df_copy
