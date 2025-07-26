#!/usr/bin/env python3
"""
Command-line interface for UnitedSystem.

This module provides a CLI for common UnitedSystem operations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from united_system._dataframe.column_key import ColumnKey
from united_system._dataframe.united_dataframe import UnitedDataframe
from united_system._units_and_dimension.unit import Unit
from united_system._scalars.real_united_scalar import RealUnitedScalar


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UnitedSystem - A library for handling physical units and united data structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert units
  united-system convert 100 m/s km/h

  # Load and display dataframe info
  united-system info data.h5

  # Convert dataframe format
  united-system convert-format data.h5 data.json
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert between units")
    convert_parser.add_argument("value", type=float, help="Numeric value")
    convert_parser.add_argument("from_unit", help="Source unit")
    convert_parser.add_argument("to_unit", help="Target unit")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display information about a file")
    info_parser.add_argument("file", help="File to analyze")
    
    # Convert format command
    format_parser = subparsers.add_parser("convert-format", help="Convert between file formats")
    format_parser.add_argument("input", help="Input file")
    format_parser.add_argument("output", help="Output file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "convert":
            convert_units(args.value, args.from_unit, args.to_unit)
        elif args.command == "info":
            show_info(args.file)
        elif args.command == "convert-format":
            convert_format(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def convert_units(value: float, from_unit: str, to_unit: str) -> None:
    """Convert a value between units."""
    try:
        source_unit = Unit(from_unit)
        target_unit = Unit(to_unit)
        
        scalar = RealUnitedScalar(value, source_unit)
        converted = scalar.to_unit(target_unit)
        
        print(f"{value} {from_unit} = {converted.value()} {to_unit}")
    except Exception as e:
        print(f"Conversion error: {e}", file=sys.stderr)
        sys.exit(1)


def show_info(file_path: str) -> None:
    """Display information about a file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    if path.suffix.lower() in [".h5", ".hdf5"]:
        show_hdf5_info(path)
    elif path.suffix.lower() == ".json":
        show_json_info(path)
    else:
        print(f"Unsupported file format: {path.suffix}", file=sys.stderr)
        sys.exit(1)


def show_hdf5_info(file_path: Path) -> None:
    """Display information about an HDF5 file."""
    try:
        df: UnitedDataframe[ColumnKey] = UnitedDataframe.from_hdf5(str(file_path), "data")
        print(f"HDF5 File: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.colkeys)}")
        print("\nColumn Information:")
        for col in df.colkeys:
            col_data = df[col]
            if hasattr(col_data, 'unit'):
                print(f"  {col}: {type(col_data).__name__} with unit {col_data.unit}")
            else:
                print(f"  {col}: {type(col_data).__name__}")
    except Exception as e:
        print(f"Error reading HDF5 file: {e}", file=sys.stderr)
        sys.exit(1)


def show_json_info(file_path: Path) -> None:
    """Display information about a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data: dict[str, Any] = json.load(f)
        
        print(f"JSON File: {file_path}")
        if isinstance(data, dict) and "columns" in data: # type: ignore
            print(f"Type: UnitedDataframe")
            print(f"Columns: {list(data['columns'].keys())}")
            print(f"Rows: {len(next(iter(data['columns'].values()))['data'])}")
        else:
            print(f"Type: Generic JSON")
            print(f"Structure: {type(data).__name__}")
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)


def convert_format(input_path: str, output_path: str) -> None:
    """Convert between file formats."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load data
        if input_file.suffix.lower() in [".h5", ".hdf5"]:
            df: UnitedDataframe[ColumnKey] = UnitedDataframe.from_hdf5(str(input_file), "data")
        elif input_file.suffix.lower() == ".json":
            df: UnitedDataframe[ColumnKey] = UnitedDataframe.from_json(input_file.read_text()) # type: ignore
        else:
            print(f"Unsupported input format: {input_file.suffix}", file=sys.stderr)
            sys.exit(1)
        
        # Save data
        if output_file.suffix.lower() in [".h5", ".hdf5"]:
            df.to_hdf5(str(output_file), "data")
        elif output_file.suffix.lower() == ".json":
            output_file.write_text(df.to_json()) # type: ignore
        else:
            print(f"Unsupported output format: {output_file.suffix}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Conversion error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 