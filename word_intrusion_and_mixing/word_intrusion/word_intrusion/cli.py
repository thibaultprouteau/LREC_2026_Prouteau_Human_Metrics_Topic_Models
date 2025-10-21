"""
Command-line interface for the word_intrusion package.
"""

import argparse
import sys
from pathlib import Path
from word_intrusion import WordIntrusionProcessor


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate word intrusion tasks from topic model data"
    )
    
    parser.add_argument(
        "input", 
        help="Input CSV file or directory containing CSV files"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory for generated tasks"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--freq-data",
        help="Path to pickle file containing word frequency data"
    )
    
    parser.add_argument(
        "--bottom-boundary",
        type=float,
        default=0.5,
        help="Bottom boundary for intruder selection (default: 0.5)"
    )
    
    parser.add_argument(
        "--top-boundary", 
        type=float,
        default=0.1,
        help="Top boundary for intruder exclusion (default: 0.1)"
    )
    
    parser.add_argument(
        "--top-words",
        type=int,
        default=4,
        help="Number of top words per topic (default: 4)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords from word pools"
    )
    
    parser.add_argument(
        "--language",
        choices=["en", "fr"],
        default="en",
        help="Language for stopword filtering (default: en)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = WordIntrusionProcessor()
    
    # Load frequency data if provided
    if args.freq_data:
        try:
            processor.load_frequencies(args.freq_data)
            print(f"Loaded frequency data from {args.freq_data}")
        except Exception as e:
            print(f"Warning: Could not load frequency data: {e}")
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Process single file
            print(f"Processing file: {input_path}")
            tasks = processor.process_csv_file(
                input_path,
                bottom_boundary=args.bottom_boundary,
                top_boundary=args.top_boundary,
                n_top_words=args.top_words,
                random_seed=args.seed,
                remove_stopwords=args.remove_stopwords,
                language=args.language
            )
            
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(exist_ok=True)
                
                import pandas as pd
                df = pd.DataFrame(tasks)
                
                if args.format == "csv":
                    output_file = output_dir / f"{input_path.stem}_tasks.csv"
                    df.to_csv(output_file, index=False)
                else:
                    output_file = output_dir / f"{input_path.stem}_tasks.json"
                    df.to_json(output_file, orient='records', indent=2)
                
                print(f"Saved {len(tasks)} tasks to {output_file}")
            else:
                print(f"Generated {len(tasks)} tasks")
                
        elif input_path.is_dir():
            # Process directory
            print(f"Processing directory: {input_path}")
            all_tasks = processor.process_directory(
                directory=input_path,
                output_dir=args.output,
                save_format=args.format,
                bottom_boundary=args.bottom_boundary,
                top_boundary=args.top_boundary,
                n_top_words=args.top_words,
                random_seed=args.seed,
                remove_stopwords=args.remove_stopwords,
                language=args.language
            )
            
            total_tasks = sum(len(tasks) for tasks in all_tasks.values())
            print(f"Processed {len(all_tasks)} files, generated {total_tasks} total tasks")
            
        else:
            print(f"Error: {input_path} is not a valid file or directory")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
