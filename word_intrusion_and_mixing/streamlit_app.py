"""
Streamlit Web Application for Word Intrusion Package

This application provides a user-friendly interface for:
1. Processing various file formats into unified format
2. Generating word intrusion tasks from processed data
"""

import streamlit as st
import pandas as pd
import json
import logging
import io
from pathlib import Path
from datetime import datetime
import time
import traceback
import shutil

# Import the word intrusion package (from local directory)
from word_intrusion import FileProcessor, WordIntrusionProcessor, load_frequency_data
from word_intrusion.topic_mixing import TopicMixingProcessor
from word_intrusion.task_selector import TaskSelector, process_word_intrusion_folder, process_mixing_folder


class StopwordLogHandler(logging.Handler):
    """Custom logging handler to capture stopword removal information for Streamlit display"""
    
    def __init__(self):
        super().__init__()
        self.logs = []
        self.stopword_stats = {}
    
    def emit(self, record):
        """Capture log records related to stopword removal"""
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
            
            # Capture logs for both old and new formats
            is_stopword_log = ("stopwords" in message or "stopword" in message) and (
                "Removed" in message or "Found" in message
            )
            
            if is_stopword_log:
                self.logs.append({
                    'level': record.levelname,
                    'message': message,
                    'timestamp': datetime.fromtimestamp(record.created)
                })
                
                # Parse new format: "Topic X: Found Y stopwords in top Z words: [word_list]"
                if "Found" in message and "stopwords in top" in message and "words:" in message:
                    try:
                        import re
                        match = re.search(r'Topic (\d+): Found (\d+) stopwords in top (\d+) words: \[(.*?)\]', message)
                        if match:
                            topic_num = int(match.group(1))
                            removed_count = int(match.group(2))
                            total_count = int(match.group(3))
                            
                            # Initialize topic stats if needed
                            if 'topics' not in self.stopword_stats:
                                self.stopword_stats['topics'] = {}
                                self.stopword_stats['total_removed'] = 0
                                self.stopword_stats['total_topics_with_stopwords'] = 0
                            
                            topic_key = f"Topic_{topic_num}"
                            self.stopword_stats['topics'][topic_key] = {
                                'removed': removed_count,
                                'total_top_words': total_count,
                                'percentage': (removed_count / total_count) * 100 if total_count > 0 else 0
                            }
                            
                            self.stopword_stats['total_removed'] += removed_count
                            if removed_count > 0:
                                self.stopword_stats['total_topics_with_stopwords'] += 1
                        
                    except (ValueError, AttributeError, IndexError):
                        pass
                
                # Parse old format: "Removed X stopwords from Y total words"
                elif "Removed" in message and "stopwords from" in message and "total words" in message:
                    try:
                        import re
                        numbers = re.findall(r'\d+', message)
                        if len(numbers) >= 2:
                            removed_count = int(numbers[0])
                            total_count = int(numbers[1])
                            
                            # Accumulate general stats
                            if 'total_removed' not in self.stopword_stats:
                                self.stopword_stats['total_removed'] = 0
                                self.stopword_stats['total_processed'] = 0
                            
                            self.stopword_stats['total_removed'] += removed_count
                            self.stopword_stats['total_processed'] += total_count
                        
                    except (ValueError, IndexError):
                        pass
    
    def get_logs(self):
        """Get captured logs"""
        return self.logs
    
    def get_stats(self):
        """Get stopword removal statistics"""
        return self.stopword_stats
    
    def clear(self):
        """Clear captured logs and stats"""
        self.logs = []
        self.stopword_stats = {}


# Initialize session state for logging
if 'stopword_log_handler' not in st.session_state:
    st.session_state.stopword_log_handler = StopwordLogHandler()

# Configure logging to use our custom handler
def setup_stopword_logging():
    """Setup logging to capture stopword removal information"""
    # Initialize the handler in session state if it doesn't exist
    if 'stopword_log_handler' not in st.session_state:
        st.session_state.stopword_log_handler = StopwordLogHandler()
    
    # Get the handler
    log_handler = st.session_state.stopword_log_handler
    log_handler.setLevel(logging.INFO)
    
    # Add handler to the core module logger
    core_logger = logging.getLogger('word_intrusion.word_intrusion.core')
    core_logger.addHandler(log_handler)
    core_logger.setLevel(logging.INFO)
    
    return log_handler


def display_stopword_stats():
    """Display stopword removal statistics in the Streamlit UI"""
    if st.session_state.stopword_log_handler.logs:
        st.subheader("📊 Stopword Removal Statistics")
        
        stats = st.session_state.stopword_log_handler.get_stats()
        logs = st.session_state.stopword_log_handler.get_logs()
        
        if stats:
            # Overall statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Stopwords Removed", stats.get('total_removed', 0))
            
            with col2:
                if 'topics' in stats:
                    st.metric("Topics with Stopwords", stats.get('total_topics_with_stopwords', 0))
                else:
                    st.metric("Total Words Processed", stats.get('total_processed', 0))
            
            with col3:
                if 'topics' in stats and stats.get('total_topics_with_stopwords', 0) > 0:
                    avg_stopwords = stats.get('total_removed', 0) / stats.get('total_topics_with_stopwords', 1)
                    st.metric("Avg Stopwords/Topic", f"{avg_stopwords:.1f}")
                else:
                    removal_rate = (stats.get('total_removed', 0) / max(stats.get('total_processed', 1), 1)) * 100
                    st.metric("Removal Rate", f"{removal_rate:.1f}%")
            
            # Topic-specific breakdown if available
            if 'topics' in stats and stats['topics']:
                st.subheader("🎯 Stopwords in Top 4 Words by Topic")
                
                topic_data = []
                for topic_name, topic_stats in stats['topics'].items():
                    topic_data.append({
                        'Topic': topic_name,
                        'Stopwords Found': topic_stats['removed'],
                        'Top Words Checked': topic_stats['total_top_words'],
                        'Percentage': f"{topic_stats['percentage']:.1f}%"
                    })
                
                if topic_data:
                    import pandas as pd
                    df = pd.DataFrame(topic_data)
                    st.dataframe(df, use_container_width=True)
        
        # Show detailed logs in an expander
        with st.expander("📝 Detailed Stopword Removal Log"):
            for log_entry in logs[-20:]:  # Show last 20 entries
                timestamp = log_entry['timestamp'].strftime("%H:%M:%S")
                st.text(f"[{timestamp}] {log_entry['message']}")



# Configure Streamlit page
st.set_page_config(
    page_title="Word Intrusion Processor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stProgress > div > div > div > div {
    background-color: #1f77b4;
}
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    margin: 1rem 0;
}
.error-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📝 Word Intrusion Processor")
    st.markdown("Process topic model files and generate word intrusion and topic mixing tasks")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Processing", "🎯 Task Generation", "🔀 Topic Mixing", "📊 Task Sampling"])
    
    with tab1:
        file_processing_tab()
    
    with tab2:
        task_generation_tab()
    
    with tab3:
        topic_mixing_tab()
    
    with tab4:
        task_sampling_tab()


def file_processing_tab():
    """First tab: File processing to unified format"""
    st.header("📁 File Processing")
    st.markdown("Convert various file formats into unified format for word intrusion tasks")
    
    # Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Selection")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["File Upload", "File Path", "Directory Path"],
            horizontal=True
        )
        
        input_path = None
        uploaded_file = None
        
        if input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=['csv', 'fuxpfx', 'json', 'txt', 'fuvp'],
                help="Supported formats: CSV, fuxpFX, JSON, TXT, fuvp"
            )
            if uploaded_file:
                # Save uploaded file temporarily
                temp_dir = Path("/tmp/word_intrusion_temp")
                temp_dir.mkdir(exist_ok=True)
                temp_file = temp_dir / uploaded_file.name
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                input_path = str(temp_file)
                
        elif input_method == "File Path":
            input_path = st.text_input(
                "Enter file path:",
                placeholder="/path/to/your/file.fuxpFX",
                help="Enter the full path to your file"
            )
            
        elif input_method == "Directory Path":
            input_path = st.text_input(
                "Enter directory path:",
                placeholder="/path/to/your/directory",
                help="Enter the full path to your directory"
            )
            
            recursive = st.checkbox(
                "Process subdirectories recursively",
                value=True,
                help="Include files in subdirectories"
            )
        
        # Output directory selection
        st.subheader("Output Settings")
        output_dir = st.text_input(
            "Output directory:",
            value="/home/tproutea/word_intrusion_output",
            help="Directory where processed files will be saved"
        )
        
        # Model name for metadata
        model_name = st.text_input(
            "Model name (optional):",
            placeholder="my_topic_model",
            help="Name to include in output filenames"
        )
        
    with col2:
        st.subheader("File Information")
        
        if input_path:
            path_obj = Path(input_path)
            if path_obj.exists():
                if path_obj.is_file():
                    st.success("✅ File found")
                    st.info(f"📄 **File:** {path_obj.name}")
                    st.info(f"📁 **Size:** {path_obj.stat().st_size / 1024:.1f} KB")
                    st.info(f"🔧 **Extension:** {path_obj.suffix}")
                elif path_obj.is_dir():
                    st.success("✅ Directory found")
                    # Count supported files
                    processor = FileProcessor()
                    supported_files = []
                    pattern = "**/*" if input_method == "Directory Path" and 'recursive' in locals() and recursive else "*"
                    for file_path in path_obj.glob(pattern):
                        if file_path.is_file() and file_path.suffix.lower() in processor.get_supported_extensions():
                            supported_files.append(file_path)
                    
                    st.info(f"📁 **Files found:** {len(supported_files)}")
                    if supported_files:
                        st.info("**File types:**")
                        extensions = {}
                        for f in supported_files:
                            ext = f.suffix.lower()
                            extensions[ext] = extensions.get(ext, 0) + 1
                        for ext, count in extensions.items():
                            st.write(f"  • {ext}: {count} files")
            else:
                st.error("❌ Path not found")
        
        # Supported formats info
        st.subheader("Supported Formats")
        st.markdown("""
        - **CSV**: Word columns + topic probabilities
        - **fuxpFX**: Custom bracket format
        - **fuvp**: Custom bracket format
        - **JSON**: Topic data in JSON format
        - **TXT**: Auto-detected text formats
        """)
    
    # Process button and results
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            if input_path and output_dir:
                process_files(input_path, output_dir, model_name, input_method == "Directory Path" and locals().get('recursive', False))
            else:
                st.error("Please provide both input path and output directory")
    
    # Display processing results
    if st.session_state.processed_files:
        st.subheader("📊 Processing Results")
        
        results_df = pd.DataFrame([
            {
                "Filename": filename,
                "Topics": len(data),
                "Status": "✅ Success"
            }
            for filename, data in st.session_state.processed_files.items()
        ])
        
        st.dataframe(results_df, use_container_width=True)


def task_generation_tab():
    """Second tab: Task generation"""
    st.header("🎯 Word Intrusion Task Generation")
    st.markdown("Generate word intrusion tasks from processed topic data")
    
    # Initialize session state
    if 'generated_tasks' not in st.session_state:
        st.session_state.generated_tasks = {}
    if 'preview_tasks' not in st.session_state:
        st.session_state.preview_tasks = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Selection")
        
        # Input method for processed files
        input_method = st.radio(
            "Choose input method:",
            ["Select Processed File", "File Path", "Directory Path"],
            horizontal=True,
            key="task_input_method"
        )
        
        input_path = None
        
        if input_method == "Select Processed File":
            if st.session_state.processed_files:
                selected_file = st.selectbox(
                    "Select a processed file:",
                    options=list(st.session_state.processed_files.keys())
                )
                if selected_file:
                    st.info(f"Selected: {selected_file}")
                    # For preview, we'll use the data directly from session state
            else:
                st.warning("No processed files available. Process files in the first tab or use file path option.")
                
        elif input_method == "File Path":
            input_path = st.text_input(
                "Enter file path:",
                placeholder="/path/to/processed/file.json",
                help="Path to a processed topic file",
                key="task_file_path"
            )
            
        elif input_method == "Directory Path":
            input_path = st.text_input(
                "Enter directory path:",
                placeholder="/path/to/processed/files",
                help="Directory containing processed topic files",
                key="task_dir_path"
            )
            
            recursive_tasks = st.checkbox(
                "Process subdirectories recursively",
                value=True,
                help="Include files in subdirectories",
                key="task_recursive"
            )
        
        # Task generation parameters
        st.subheader("Task Parameters")
        
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            n_top_words = st.slider(
                "Number of top words per task:",
                min_value=3,
                max_value=6,
                value=4,
                help="Number of coherent words in each task"
            )
            
            # Bottom boundary parameter options
            st.write("**Bottom Boundary (Intruder Pool):**")
            bottom_boundary_type = st.radio(
                "Boundary type:",
                ["Fraction (0-1)", "Number of words", "Range [start, end]"],
                key="bottom_boundary_type",
                help="Choose how to specify the bottom boundary for intruder candidates"
            )
            
            if bottom_boundary_type == "Fraction (0-1)":
                bottom_boundary = st.slider(
                    "Fraction of words:",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Fraction of words to consider as intruder candidates"
                )
            elif bottom_boundary_type == "Number of words":
                bottom_boundary_n = st.number_input(
                    "Number of words from end:",
                    min_value=2,
                    max_value=1000,
                    value=100,
                    help="Number of words to take from the end of the topic list"
                )
                bottom_boundary = [bottom_boundary_n]
            else:  # Range
                col_start, col_end = st.columns(2)
                with col_start:
                    bottom_start = st.number_input(
                        "Start index:",
                        min_value=0,
                        max_value=1000,
                        value=50,
                        help="Start index for word range"
                    )
                with col_end:
                    bottom_end = st.number_input(
                        "End index:",
                        min_value=1,
                        max_value=1000,
                        value=200,
                        help="End index for word range"
                    )
                if bottom_end <= bottom_start:
                    st.warning("⚠️ End index must be greater than start index")
                    bottom_boundary = [bottom_start, bottom_start + 1]  # Fallback
                else:
                    bottom_boundary = [bottom_start, bottom_end]
            
        with col_param2:
            # Top boundary parameter options
            st.write("**Top Boundary (Exclusion):**")
            top_boundary_type = st.radio(
                "Boundary type:",
                ["Fraction (0-1)", "Number of words", "Range [start, end]"],
                key="top_boundary_type",
                help="Choose how to specify the top boundary for exclusion"
            )
            
            if top_boundary_type == "Fraction (0-1)":
                top_boundary = st.slider(
                    "Fraction of words:",
                    min_value=0.05,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="Fraction of top words to exclude from intruders"
                )
            elif top_boundary_type == "Number of words":
                top_boundary_n = st.number_input(
                    "Number of words from start:",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Number of words to take from the start of the topic list"
                )
                top_boundary = [top_boundary_n]
            else:  # Range
                col_start_top, col_end_top = st.columns(2)
                with col_start_top:
                    top_start = st.number_input(
                        "Start index:",
                        min_value=0,
                        max_value=100,
                        value=0,
                        help="Start index for word range",
                        key="top_start"
                    )
                with col_end_top:
                    top_end = st.number_input(
                        "End index:",
                        min_value=1,
                        max_value=100,
                        value=10,
                        help="End index for word range",
                        key="top_end"
                    )
                if top_end <= top_start:
                    st.warning("⚠️ End index must be greater than start index")
                    top_boundary = [top_start, top_start + 1]  # Fallback
                else:
                    top_boundary = [top_start, top_end]
            
            random_seed = st.number_input(
                "Random seed:",
                min_value=1,
                max_value=9999,
                value=42,
                help="Seed for reproducible results"
            )
        
        # Frequency data option
        use_frequency_data = st.checkbox(
            "Use frequency data for better intruder selection",
            help="Improves intruder quality by considering word frequencies"
        )
        
        freq_file_path = None
        if use_frequency_data:
            freq_file_path = st.text_input(
                "Frequency data file path:",
                placeholder="/path/to/frequency_data.pkl",
                help="Path to pickle file containing word frequencies"
            )
        
        # Display boundary parameter summary
        st.divider()
        st.subheader("📋 Parameter Summary")
        
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.write("**Bottom Boundary (Intruder Pool):**")
            if isinstance(bottom_boundary, float):
                st.info(f"Using {bottom_boundary:.1%} of words from the end")
            elif isinstance(bottom_boundary, list) and len(bottom_boundary) == 1:
                st.info(f"Using last {bottom_boundary[0]} words")
            elif isinstance(bottom_boundary, list) and len(bottom_boundary) == 2:
                st.info(f"Using words from index {bottom_boundary[0]} to {bottom_boundary[1]}")
        
        with col_summary2:
            st.write("**Top Boundary (Exclusion):**")
            if isinstance(top_boundary, float):
                st.info(f"Excluding top {top_boundary:.1%} of words")
            elif isinstance(top_boundary, list) and len(top_boundary) == 1:
                st.info(f"Excluding first {top_boundary[0]} words")
            elif isinstance(top_boundary, list) and len(top_boundary) == 2:
                st.info(f"Using words from index {top_boundary[0]} to {top_boundary[1]}")
    
    with col2:
        st.subheader("Generation Settings")
        
        # Output format
        output_format = st.selectbox(
            "Output format:",
            ["CSV", "JSON"],
            help="Format for saving generated tasks"
        )
        
        # Stopword filtering options
        st.divider()
        st.write("**Stopword Filtering:**")
        remove_stopwords = st.checkbox(
            "Remove stopwords from word pools",
            value=False,
            help="Remove common stopwords from both top and bottom word pools to improve task quality"
        )
        
        language = st.selectbox(
            "Language for stopwords:",
            ["en", "fr"],
            index=0,
            help="Select language for stopword filtering (English or French)",
            disabled=not remove_stopwords
        )
        
        if remove_stopwords:
            st.info("ℹ️ Stopwords will be removed from both top and bottom pools using spaCy")
        else:
            st.info("ℹ️ All words will be kept in the pools")
        
        # Preview settings
        preview_limit = st.slider(
            "Preview limit:",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of tasks to show in preview"
        )
    
    # Preview button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("👀 Preview Tasks", use_container_width=True):
            preview_tasks(
                input_method, 
                input_path, 
                locals().get('selected_file'),
                n_top_words,
                bottom_boundary,
                top_boundary,
                random_seed,
                freq_file_path,
                preview_limit,
                remove_stopwords,
                language
            )
    
    # Display preview
    if st.session_state.preview_tasks:
        st.subheader("👀 Task Preview")
        
        for i, task in enumerate(st.session_state.preview_tasks, 1):
            with st.expander(f"Task {i}: {task.get('model', 'Unknown Model')}"):
                words = [task[f'word{j}'] for j in range(1, 6)]
                
                st.write("**Words:**", " | ".join(words))
                st.write("**Intruder:**", task['intruder'])
                st.write("**Task ID:**", task['text'])
                
                # Highlight the intruder
                highlighted_words = []
                for word in words:
                    if word == task['intruder']:
                        highlighted_words.append(f"🔴 **{word}**")
                    else:
                        highlighted_words.append(f"🟢 {word}")
                
                st.markdown("**Visual:** " + " | ".join(highlighted_words))
    
    # Output settings and generation
    st.divider()
    
    output_dir_tasks = st.text_input(
        "Output directory for tasks:",
        value="/home/tproutea/word_intrusion_tasks",
        help="Directory where generated tasks will be saved"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎯 Generate and Save Tasks", type="primary", use_container_width=True):
            if (input_method == "Select Processed File" and 'selected_file' in locals() and selected_file) or input_path:
                generate_and_save_tasks(
                    input_method,
                    input_path,
                    locals().get('selected_file'),
                    output_dir_tasks,
                    output_format,
                    n_top_words,
                    bottom_boundary,
                    top_boundary,
                    random_seed,
                    freq_file_path,
                    locals().get('recursive_tasks', False),
                    remove_stopwords,
                    language
                )
            else:
                st.error("Please select input data")
    
    # Display generation results
    if st.session_state.generated_tasks:
        st.subheader("📊 Generation Results")
        
        results_df = pd.DataFrame([
            {
                "Model": model_name,
                "Tasks Generated": len(tasks),
                "Status": "✅ Success"
            }
            for model_name, tasks in st.session_state.generated_tasks.items()
        ])
        
        st.dataframe(results_df, use_container_width=True)


def process_files(input_path, output_dir, model_name, recursive=False):
    """Process files and save to unified format"""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize file processor
        file_processor = FileProcessor()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            # Process single file
            status_text.text("Processing single file...")
            progress_bar.progress(25)
            
            topics_data = file_processor.process_file(input_path)
            progress_bar.progress(75)
            
            # Generate filename with timestamp and model name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = input_path_obj.stem
            if model_name:
                filename = f"{model_name}_{base_name}_{timestamp}.json"
            else:
                filename = f"{base_name}_{timestamp}.json"
            
            # Save processed data
            output_file = output_path / filename
            with open(output_file, 'w') as f:
                json.dump(topics_data, f, indent=2)
            
            # Store in session state
            st.session_state.processed_files[filename] = topics_data
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            st.success(f"Successfully processed 1 file. Saved to: {output_file}")
            
        elif input_path_obj.is_dir():
            # Process directory
            status_text.text("Scanning directory...")
            progress_bar.progress(10)
            
            # Get all supported files
            supported_files = []
            pattern = "**/*" if recursive else "*"
            for file_path in input_path_obj.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in file_processor.get_supported_extensions():
                    supported_files.append(file_path)
            
            if not supported_files:
                st.warning("No supported files found in the directory")
                return
            
            progress_bar.progress(20)
            
            # Process each file
            processed_count = 0
            total_files = len(supported_files)
            
            for i, file_path in enumerate(supported_files):
                try:
                    status_text.text(f"Processing {file_path.name}... ({i+1}/{total_files})")
                    
                    topics_data = file_processor.process_file(file_path)
                    
                    # Generate filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = file_path.stem
                    if model_name:
                        filename = f"{model_name}_{base_name}_{timestamp}.json"
                    else:
                        filename = f"{base_name}_{timestamp}.json"
                    
                    # Save processed data
                    output_file = output_path / filename
                    with open(output_file, 'w') as f:
                        json.dump(topics_data, f, indent=2)
                    
                    # Store in session state
                    st.session_state.processed_files[filename] = topics_data
                    
                    processed_count += 1
                    
                    # Update progress
                    progress = 20 + (i + 1) / total_files * 80
                    progress_bar.progress(int(progress))
                    
                except Exception as e:
                    st.warning(f"Error processing {file_path.name}: {str(e)}")
                    continue
            
            status_text.text("✅ Processing complete!")
            st.success(f"Successfully processed {processed_count}/{total_files} files. Saved to: {output_path}")
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.error(traceback.format_exc())


def preview_tasks(input_method, input_path, selected_file, n_top_words, bottom_boundary, top_boundary, random_seed, freq_file_path, preview_limit, remove_stopwords, language):
    """Preview word intrusion tasks"""
    try:
        # Setup logging for stopword removal
        if remove_stopwords:
            setup_stopword_logging()
            st.session_state.stopword_log_handler.clear()
        
        # Initialize processor
        processor = WordIntrusionProcessor()
        
        # Load frequency data if provided
        if freq_file_path and Path(freq_file_path).exists():
            try:
                frequency_data = load_frequency_data(freq_file_path)
                processor.set_frequency_data(frequency_data)
                st.info("✅ Frequency data loaded for better intruder selection")
            except Exception as e:
                st.warning(f"Could not load frequency data: {str(e)}")
        
        # Get topics data
        topics_data = None
        
        if input_method == "Select Processed File" and selected_file:
            topics_data = st.session_state.processed_files[selected_file]
            model_name = selected_file
            
        elif input_path:
            # Try to process the file/directory
            file_processor = FileProcessor()
            if Path(input_path).is_file():
                topics_data = file_processor.process_file(input_path)
                model_name = Path(input_path).stem
            else:
                st.error("Directory preview not implemented. Please select a specific file.")
                return
        
        if topics_data:
            # Generate tasks
            tasks = processor.process_topics(
                topics_data=topics_data,
                model_name=model_name,
                bottom_boundary=bottom_boundary,
                top_boundary=top_boundary,
                n_top_words=n_top_words,
                random_seed=random_seed,
                remove_stopwords=remove_stopwords,
                language=language
            )
            
            # Store preview tasks (limited)
            st.session_state.preview_tasks = tasks[:preview_limit]
            
            st.info(f"Preview showing {len(st.session_state.preview_tasks)} out of {len(tasks)} total tasks")
            
            # Display stopword removal statistics if enabled
            if remove_stopwords:
                display_stopword_stats()
        
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")
        st.error(traceback.format_exc())


def generate_and_save_tasks(input_method, input_path, selected_file, output_dir, output_format, n_top_words, bottom_boundary, top_boundary, random_seed, freq_file_path, recursive=False, remove_stopwords=False, language='en'):
    """Generate and save word intrusion tasks"""
    try:
        # Setup logging for stopword removal
        if remove_stopwords:
            setup_stopword_logging()
            st.session_state.stopword_log_handler.clear()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        processor = WordIntrusionProcessor()
        
        # Load frequency data if provided
        if freq_file_path and Path(freq_file_path).exists():
            try:
                frequency_data = load_frequency_data(freq_file_path)
                processor.set_frequency_data(frequency_data)
            except Exception as e:
                st.warning(f"Could not load frequency data: {str(e)}")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if input_method == "Select Processed File" and selected_file:
            # Process single selected file
            status_text.text("Generating tasks...")
            progress_bar.progress(25)
            
            topics_data = st.session_state.processed_files[selected_file]
            model_name = Path(selected_file).stem
            
            tasks = processor.process_topics(
                topics_data=topics_data,
                model_name=model_name,
                bottom_boundary=bottom_boundary,
                top_boundary=top_boundary,
                n_top_words=n_top_words,
                random_seed=random_seed,
                remove_stopwords=remove_stopwords,
                language=language
            )
            
            progress_bar.progress(75)
            
            # Save tasks
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if output_format.lower() == "csv":
                output_file = output_path / f"{model_name}_tasks_{timestamp}.csv"
                df = pd.DataFrame(tasks)
                df.to_csv(output_file, index=False)
            else:
                output_file = output_path / f"{model_name}_tasks_{timestamp}.json"
                with open(output_file, 'w') as f:
                    json.dump(tasks, f, indent=2)
            
            st.session_state.generated_tasks[model_name] = tasks
            
            progress_bar.progress(100)
            status_text.text("✅ Generation complete!")
            
            st.success(f"Generated {len(tasks)} tasks. Saved to: {output_file}")
            
            # Display stopword removal statistics if enabled
            if remove_stopwords:
                display_stopword_stats()
            
        elif input_path:
            # Process from file path
            input_path_obj = Path(input_path)
            
            if input_path_obj.is_file():
                # Single file
                status_text.text("Processing file and generating tasks...")
                progress_bar.progress(25)
                
                tasks = processor.process_file(
                    file_path=input_path,
                    bottom_boundary=bottom_boundary,
                    top_boundary=top_boundary,
                    n_top_words=n_top_words,
                    random_seed=random_seed,
                    remove_stopwords=remove_stopwords,
                    language=language
                )
                
                progress_bar.progress(75)
                
                # Save tasks
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = input_path_obj.stem
                
                if output_format.lower() == "csv":
                    output_file = output_path / f"{model_name}_tasks_{timestamp}.csv"
                    df = pd.DataFrame(tasks)
                    df.to_csv(output_file, index=False)
                else:
                    output_file = output_path / f"{model_name}_tasks_{timestamp}.json"
                    with open(output_file, 'w') as f:
                        json.dump(tasks, f, indent=2)
                
                st.session_state.generated_tasks[model_name] = tasks
                
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                
                st.success(f"Generated {len(tasks)} tasks. Saved to: {output_file}")
                
                # Display stopword removal statistics if enabled
                if remove_stopwords:
                    display_stopword_stats()
                
            elif input_path_obj.is_dir():
                # Directory processing
                all_tasks = processor.process_directory(
                    directory=input_path,
                    output_dir=None,  # We'll save manually
                    save_format=output_format.lower(),
                    bottom_boundary=bottom_boundary,
                    top_boundary=top_boundary,
                    n_top_words=n_top_words,
                    random_seed=random_seed,
                    recursive=recursive,
                    remove_stopwords=remove_stopwords,
                    language=language
                )
                
                progress_bar.progress(75)
                
                # Save all tasks
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                total_tasks = 0
                
                for model_name, tasks in all_tasks.items():
                    if output_format.lower() == "csv":
                        output_file = output_path / f"{Path(model_name).stem}_tasks_{timestamp}.csv"
                        df = pd.DataFrame(tasks)
                        df.to_csv(output_file, index=False)
                    else:
                        output_file = output_path / f"{Path(model_name).stem}_tasks_{timestamp}.json"
                        with open(output_file, 'w') as f:
                            json.dump(tasks, f, indent=2)
                    
                    total_tasks += len(tasks)
                
                st.session_state.generated_tasks.update(all_tasks)
                
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                
                st.success(f"Generated {total_tasks} tasks from {len(all_tasks)} files. Saved to: {output_path}")
                
                # Display stopword removal statistics if enabled
                if remove_stopwords:
                    display_stopword_stats()
        
    except Exception as e:
        st.error(f"Error during task generation: {str(e)}")
        st.error(traceback.format_exc())


def _create_topic_mixing_processor(model_name: str) -> 'TopicMixingProcessor':
    """Create TopicMixingProcessor with correct trust_remote_code setting based on model"""
    trust_remote_code = "NovaSearch" in model_name or "stella" in model_name
    return TopicMixingProcessor(model_name=model_name, trust_remote_code=trust_remote_code)


def topic_mixing_tab():
    """Third tab: Topic mixing task generation"""
    st.header("🔀 Topic Mixing")
    st.markdown("Generate topic mixing tasks using closest semantic topic pairs for improved coherence")
    
    # Initialize session state
    if 'mixing_tasks' not in st.session_state:
        st.session_state.mixing_tasks = {}
    if 'mixing_preview' not in st.session_state:
        st.session_state.mixing_preview = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Selection")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Select Processed File", "File Path", "Directory Path"],
            horizontal=True,
            key="mixing_input_method"
        )
        
        input_path = None
        selected_file = None
        
        if input_method == "Select Processed File":
            if st.session_state.processed_files:
                selected_file = st.selectbox(
                    "Select a processed file:",
                    options=list(st.session_state.processed_files.keys()),
                    key="mixing_selected_file"
                )
                if selected_file:
                    st.info(f"Selected: {selected_file}")
                    # Display file info
                    topics_data = st.session_state.processed_files[selected_file]
                    st.info(f"📊 **Topics:** {len(topics_data)}")
            else:
                st.warning("No processed files available. Process files in the first tab or use file path option.")
                
        elif input_method == "File Path":
            input_path = st.text_input(
                "Enter file path:",
                placeholder="/path/to/processed/file.json",
                help="Path to a processed topic file (JSON format from Tab 1)",
                key="mixing_file_path"
            )
            
            if input_path:
                path_obj = Path(input_path)
                if path_obj.exists() and path_obj.is_file():
                    st.success("✅ File found")
                    st.info(f"📄 **File:** {path_obj.name}")
                    st.info(f"📁 **Size:** {path_obj.stat().st_size / 1024:.1f} KB")
                    st.info(f"🔧 **Extension:** {path_obj.suffix}")
                else:
                    st.error("❌ File not found")
            
        elif input_method == "Directory Path":
            input_path = st.text_input(
                "Enter directory path:",
                placeholder="/path/to/processed/files",
                help="Directory containing processed topic files (JSON format)",
                key="mixing_dir_path"
            )
            
            recursive = st.checkbox(
                "Process subdirectories recursively",
                value=True,
                help="Include files in subdirectories",
                key="mixing_recursive"
            )
            
            if input_path:
                path_obj = Path(input_path)
                if path_obj.exists() and path_obj.is_dir():
                    st.success("✅ Directory found")
                    # Count supported files
                    supported_files = []
                    pattern = "**/*" if recursive else "*"
                    for file_path in path_obj.glob(pattern):
                        if file_path.is_file() and file_path.suffix.lower() in ['.json']:
                            supported_files.append(file_path)
                    
                    st.info(f"📁 **Files found:** {len(supported_files)}")
                    if supported_files:
                        st.info("**File types:**")
                        extensions = {}
                        for f in supported_files:
                            ext = f.suffix.lower()
                            extensions[ext] = extensions.get(ext, 0) + 1
                        for ext, count in extensions.items():
                            st.write(f"  • {ext}: {count} files")
                else:
                    st.error("❌ Directory not found")
    
    with col2:
        st.subheader("Mixing Parameters")
        
        # Number of words per task (must be multiple of 2)
        words_per_task = st.number_input(
            "Words per task:",
            min_value=4,
            max_value=20,
            value=10,
            step=2,
            help="Total number of words in each mixing task (must be even)",
            key="mixing_words_per_task"
        )
        
        # Calculate words per topic
        words_per_topic = words_per_task // 2
        st.info(f"Words per topic: {words_per_topic}")
        
        # Supported formats info
        st.subheader("Input Format")
        st.markdown("""
        - **JSON**: Processed files from Tab 1
        - **Format**: List of topics with word-probability pairs
        - **Source**: Use files generated in File Processing tab
        """)
        
        # New approach explanation
        st.subheader("🎯 Closest-Topic Approach")
        st.markdown("""
        - **Semantic Pairing**: Topics are paired with their most similar neighbor
        - **Single-topic Tasks**: Words from same topic (50% of tasks)
        - **Mixed-topic Tasks**: Words from closest topic pairs (50% of tasks)
        - **Quality**: Ensures meaningful semantic relationships in mixed tasks
        """)
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            top_n = st.number_input(
                "Words for similarity computation:",
                min_value=10,
                max_value=200,
                value=50,
                help="Number of top words to use for computing topic similarities",
                key="mixing_top_n"
            )
            
            model_name = st.selectbox(
                "Embedding model:",
                [
                    "intfloat/e5-small-v2",
                    "BAAI/bge-small-en-v1.5", 
                    "thenlper/gte-small",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "NovaSearch/stella_en_1.5B_v5"
                ],
                index=0,
                help="Sentence transformer model for computing topic similarities",
                key="mixing_model"
            )
            
            remove_stopwords = st.checkbox(
                "Remove stopwords",
                value=False,
                help="Filter out common stopwords from topics",
                key="mixing_stopwords"
            )
            
            language = st.selectbox(
                "Language:",
                ["en", "fr"],
                index=0,
                help="Language for stopword filtering",
                key="mixing_language"
            )
            
            random_seed = st.number_input(
                "Random seed:",
                min_value=1,
                max_value=999999,
                value=42,
                help="Seed for reproducible results",
                key="mixing_seed"
            )
    
    # Output settings (for single file processing)
    st.divider()
    
    if input_method == "Select Processed File" or input_method == "File Path":
        output_dir_mixing = st.text_input(
            "Output directory for mixing tasks:",
            value="/home/tproutea/word_intrusion_mixing_tasks",
            help="Directory where generated mixing tasks will be saved"
        )
    
    # Processing section
    st.divider()
    
    if input_method == "Select Processed File" or input_method == "File Path":
        # Single file processing
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔍 Preview Tasks", type="secondary", use_container_width=True, key="mixing_preview_btn"):
                if input_method == "Select Processed File" and selected_file:
                    preview_mixing_tasks_from_data(selected_file, st.session_state.processed_files[selected_file], words_per_topic, top_n, model_name, remove_stopwords, language, random_seed)
                elif input_method == "File Path" and input_path:
                    preview_mixing_tasks(input_path, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed)
                else:
                    st.error("Please select or provide input data")
        
        with col2:
            if st.button("🚀 Generate Tasks", type="primary", use_container_width=True, key="mixing_generate_btn"):
                if input_method == "Select Processed File" and selected_file:
                    generate_mixing_tasks_from_data(selected_file, st.session_state.processed_files[selected_file], words_per_topic, top_n, model_name, remove_stopwords, language, random_seed)
                elif input_method == "File Path" and input_path:
                    generate_mixing_tasks(input_path, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed)
                else:
                    st.error("Please select or provide input data")
        
        with col3:
            if st.session_state.mixing_tasks:
                # Output format selection (similar to word intrusion)
                output_format_mixing = st.selectbox(
                    "Output format:",
                    ["CSV", "JSON"],
                    help="Format for saving generated mixing tasks",
                    key="mixing_output_format"
                )
                
                if st.button("💾 Save Tasks", use_container_width=True, key="mixing_save_btn"):
                    save_mixing_tasks(output_dir_mixing, output_format_mixing)
    
    elif input_method == "Directory Path":
        # Directory processing
        st.subheader("Batch Processing Settings")
        
        col_dir1, col_dir2 = st.columns(2)
        
        with col_dir1:
            output_dir = st.text_input(
                "Output directory:",
                placeholder="/path/to/output/directory",
                help="Directory to save mixing task files",
                key="mixing_output_dir"
            )
        
        with col_dir2:
            output_format_batch = st.selectbox(
                "Output format:",
                ["CSV", "JSON"],
                help="Format for saving batch mixing tasks",
                key="mixing_batch_output_format"
            )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Process Directory", type="primary", use_container_width=True, key="mixing_batch_btn"):
                if input_path and output_dir:
                    process_mixing_directory(input_path, output_dir, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed, recursive, output_format_batch)
                else:
                    st.error("Please provide both input and output directories")
    
    # Display preview or results
    if st.session_state.mixing_preview is not None:
        st.subheader("Preview Results")
        display_mixing_preview(st.session_state.mixing_preview)
    
    if st.session_state.mixing_tasks:
        st.subheader("Generated Tasks")
        display_mixing_results(st.session_state.mixing_tasks)


def preview_mixing_tasks(file_path, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed):
    """Preview mixing tasks without generating all of them"""
    try:
        # Setup logging for stopword removal
        if remove_stopwords:
            setup_stopword_logging()
            st.session_state.stopword_log_handler.clear()
            
        with st.spinner("Computing topic similarities and generating preview..."):
            processor = _create_topic_mixing_processor(model_name)
            
            # Generate a limited set of tasks for preview
            mixing_tasks = processor.process_file_mixing(
                file_path,
                top_n=top_n,
                mixing_n_tops=words_per_topic,
                remove_stopwords=remove_stopwords,
                language=language,
                random_seed=random_seed,
                show_progress=True
            )
            
            # Store preview (first 5 tasks)
            st.session_state.mixing_preview = mixing_tasks[:5] if len(mixing_tasks) > 5 else mixing_tasks
            
            st.success(f"Preview generated! Showing {len(st.session_state.mixing_preview)} sample tasks out of {len(mixing_tasks)} total.")
            
            # Display stopword removal statistics if enabled
            if remove_stopwords:
                display_stopword_stats()
    
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")
        if st.checkbox("Show detailed error", key="mixing_preview_error"):
            st.error(traceback.format_exc())


def generate_mixing_tasks(file_path, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed):
    """Generate all mixing tasks for a single file"""
    try:
        with st.spinner("Generating mixing tasks..."):
            processor = _create_topic_mixing_processor(model_name)
            
            mixing_tasks = processor.process_file_mixing(
                file_path,
                top_n=top_n,
                mixing_n_tops=words_per_topic,
                remove_stopwords=remove_stopwords,
                language=language,
                random_seed=random_seed,
                show_progress=True
            )
            
            st.session_state.mixing_tasks = {Path(file_path).stem: mixing_tasks}
            
            st.success(f"Generated {len(mixing_tasks)} mixing tasks!")
    
    except Exception as e:
        st.error(f"Error generating tasks: {str(e)}")
        if st.checkbox("Show detailed error", key="mixing_generate_error"):
            st.error(traceback.format_exc())


def preview_mixing_tasks_from_data(filename, topics_data, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed):
    """Preview mixing tasks from processed data in session state"""
    try:
        with st.spinner("Computing topic similarities and generating preview..."):
            processor = _create_topic_mixing_processor(model_name)
            
            # Generate a limited set of tasks for preview
            mixing_tasks = processor.process_mixing_tasks(
                topics_data=topics_data,
                top_n=top_n,
                mixing_n_tops=words_per_topic,
                remove_stopwords=remove_stopwords,
                language=language,
                random_seed=random_seed,
                show_progress=False
            )
            
            # Store preview (first 5 tasks)
            st.session_state.mixing_preview = mixing_tasks[:5] if len(mixing_tasks) > 5 else mixing_tasks
            
            st.success(f"Preview generated! Showing {len(st.session_state.mixing_preview)} sample tasks out of {len(mixing_tasks)} total.")
    
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")
        if st.checkbox("Show detailed error", key="mixing_preview_data_error"):
            st.error(traceback.format_exc())


def generate_mixing_tasks_from_data(filename, topics_data, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed):
    """Generate all mixing tasks from processed data in session state"""
    try:
        with st.spinner("Generating mixing tasks..."):
            processor = _create_topic_mixing_processor(model_name)
            
            mixing_tasks = processor.process_mixing_tasks(
                topics_data=topics_data,
                top_n=top_n,
                mixing_n_tops=words_per_topic,
                remove_stopwords=remove_stopwords,
                language=language,
                random_seed=random_seed,
                show_progress=False
            )
            
            st.session_state.mixing_tasks = {Path(filename).stem: mixing_tasks}
            
            st.success(f"Generated {len(mixing_tasks)} mixing tasks!")
    
    except Exception as e:
        st.error(f"Error generating tasks: {str(e)}")
        if st.checkbox("Show detailed error", key="mixing_generate_data_error"):
            st.error(traceback.format_exc())


def process_mixing_directory(input_dir, output_dir, words_per_topic, top_n, model_name, remove_stopwords, language, random_seed, recursive, output_format="CSV"):
    """Process multiple files in a directory"""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with st.spinner("Processing directory..."):
            processor = _create_topic_mixing_processor(model_name)
            
            # Get all supported files
            directory_path = Path(input_dir)
            pattern = "**/*" if recursive else "*"
            supported_files = []
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in ['.json']:
                    supported_files.append(file_path)
            
            if not supported_files:
                st.error("No supported JSON files found in the directory")
                return
            
            # Process each file
            total_tasks = 0
            processed_files = []
            failed_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(supported_files):
                try:
                    status_text.text(f"Processing {file_path.name}... ({i+1}/{len(supported_files)})")
                    
                    # Generate mixing tasks for this file
                    mixing_tasks = processor.process_file_mixing(
                        file_path,
                        top_n=top_n,
                        mixing_n_tops=words_per_topic,
                        remove_stopwords=remove_stopwords,
                        language=language,
                        random_seed=random_seed,
                        show_progress=False
                    )
                    
                    # Save tasks using same naming convention as word intrusion
                    model_name_file = file_path.stem
                    if output_format.lower() == "csv":
                        output_file = output_path / f"{model_name_file}_mixing_tasks_{timestamp}.csv"
                        df = pd.DataFrame(mixing_tasks)
                        df.to_csv(output_file, index=False)
                    else:
                        output_file = output_path / f"{model_name_file}_mixing_tasks_{timestamp}.json"
                        with open(output_file, 'w') as f:
                            json.dump(mixing_tasks, f, indent=2)
                    
                    processed_files.append(str(file_path))
                    total_tasks += len(mixing_tasks)
                    
                    # Update progress
                    progress = (i + 1) / len(supported_files) * 100
                    progress_bar.progress(int(progress))
                    
                except Exception as e:
                    failed_files.append({'file': str(file_path), 'error': str(e)})
                    continue
            
            progress_bar.progress(100)
            status_text.text("✅ Batch processing complete!")
            
            st.success(f"Batch processing complete!")
            st.info(f"✅ Processed: {len(processed_files)} files")
            st.info(f"❌ Failed: {len(failed_files)} files")
            st.info(f"📊 Total tasks: {total_tasks}")
            st.info(f"💾 Saved to: {output_path}")
            
            if failed_files:
                st.error("Failed files:")
                for failed in failed_files:
                    st.write(f"• {failed['file']}: {failed['error']}")
    
    except Exception as e:
        st.error(f"Error processing directory: {str(e)}")
        if st.checkbox("Show detailed error", key="mixing_batch_error"):
            st.error(traceback.format_exc())


def save_mixing_tasks(output_dir, output_format="csv"):
    """Save generated mixing tasks to file with proper naming convention"""
    try:
        if st.session_state.mixing_tasks:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get the tasks and model name
            model_name, tasks = list(st.session_state.mixing_tasks.items())[0]
            
            # Generate timestamp and filename using same format as word intrusion
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format.lower() == "csv":
                output_file = output_path / f"{model_name}_mixing_tasks_{timestamp}.csv"
                # Save as CSV using pandas
                df = pd.DataFrame(tasks)
                df.to_csv(output_file, index=False)
            else:
                output_file = output_path / f"{model_name}_mixing_tasks_{timestamp}.json"
                # Save as JSON
                with open(output_file, 'w') as f:
                    json.dump(tasks, f, indent=2)
            
            st.success(f"Saved {len(tasks)} mixing tasks to: {output_file}")
        else:
            st.error("No tasks to save")
    
    except Exception as e:
        st.error(f"Error saving tasks: {str(e)}")
        if st.checkbox("Show detailed save error", key="mixing_save_error_detail"):
            st.error(traceback.format_exc())


def display_mixing_preview(preview_tasks):
    """Display preview of mixing tasks"""
    for i, task in enumerate(preview_tasks):
        task_type = "Single-topic" if task['quartile'] == -1 else "Mixed-topic (Closest)"
        similarity_text = f"(Similarity: {task['similarity']:.3f})" if task['quartile'] != -1 else "(Same topic)"
        
        with st.expander(f"Task {i+1}: {task['task_id']} - {task_type} {similarity_text}"):
            col1, col2 = st.columns(2)
            
            with col1:
                if task['quartile'] == -1:
                    st.write("**Single topic words (first half):**")
                    st.write(", ".join(task['topic1_words']))
                    
                    st.write("**Single topic words (second half):**")
                    st.write(", ".join(task['topic2_words']))
                else:
                    st.write("**Topic 1 words:**")
                    st.write(", ".join(task['topic1_words']))
                    
                    st.write("**Topic 2 words (closest):**")
                    st.write(", ".join(task['topic2_words']))
            
            with col2:
                st.write("**All words:**")
                st.write(", ".join(task['mixed_words']))
                
                if task['quartile'] == -1:
                    st.write("**Task type:** Single-topic (control)")
                    st.write(f"**Topic:** {task['mixed_topics'][0]}")
                else:
                    st.write("**Type:** Closest-topic mixing")
                    st.write(f"**Mixed topics:** {task['mixed_topics'][0]} + {task['mixed_topics'][1]} (closest pair)")
                    st.write(f"**Similarity:** {task['similarity']:.3f}")
                
                # Display HTML preview of mixed_set
                st.write("**Visual preview:**")
                st.markdown(task['mixed_set'], unsafe_allow_html=True)


def display_mixing_results(all_tasks):
    """Display results of mixing task generation"""
    for filename, tasks in all_tasks.items():
        st.write(f"**{filename}:** {len(tasks)} tasks")
        
        # Separate single-topic and mixed-topic tasks
        single_tasks = [task for task in tasks if task['quartile'] == -1]
        mixed_tasks = [task for task in tasks if task['quartile'] != -1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tasks", len(tasks))
        with col2:
            st.metric("Single-topic", len(single_tasks))
        with col3:
            st.metric("Closest-topic Mixed", len(mixed_tasks))
        with col4:
            if mixed_tasks:
                similarities = [task['similarity'] for task in mixed_tasks]
                st.metric("Avg Similarity", f"{sum(similarities)/len(similarities):.3f}")
            else:
                st.metric("Avg Similarity", "N/A")
        
        # Show detailed statistics for mixed tasks
        if mixed_tasks:
            similarities = [task['similarity'] for task in mixed_tasks]
            st.write("**Closest-topic similarity range:**")
            col_min, col_max = st.columns(2)
            with col_min:
                st.write(f"Min: {min(similarities):.3f}")
            with col_max:
                st.write(f"Max: {max(similarities):.3f}")
        
        # Task distribution
        st.write("**Task distribution:**")
        st.write(f"• Single-topic (Q=-1): {len(single_tasks)} tasks")
        if mixed_tasks:
            st.write(f"• Closest-topic mixed (Q=0): {len(mixed_tasks)} tasks")
        
        # Show balance
        if tasks:
            single_pct = len(single_tasks) / len(tasks) * 100
            mixed_pct = len(mixed_tasks) / len(tasks) * 100
            st.write("**Task balance:**")
            st.write(f"• Single-topic: {single_pct:.1f}%")
            st.write(f"• Closest-topic mixed: {mixed_pct:.1f}%")


def task_sampling_tab():
    """Fourth tab: Task sampling for human evaluation"""
    st.header("📊 Task Sampling")
    st.markdown("Sample tasks for human evaluation with representative coverage")
    
    # Initialize session state
    if 'sampling_preview' not in st.session_state:
        st.session_state.sampling_preview = None
    if 'sampling_results' not in st.session_state:
        st.session_state.sampling_results = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Selection")
        
        # Task type selection
        task_type = st.selectbox(
            "Task type:",
            ["Word Intrusion", "Topic Mixing"],
            help="Select the type of tasks to sample"
        )
        
        # Show sampling strategy info for mixing tasks
        if task_type == "Topic Mixing":
            st.info("""
            **📊 Mixing Task Sampling Strategy:**
            - **50% Single-topic (Q=-1)**: Tasks with words from the same coherent topic
            - **50% Mixed-topic (Q=0)**: Tasks with words from closest semantic topic pairs
            - This ensures balanced evaluation between coherent and mixed content
            """)    
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["File Path", "Folder Path"],
            horizontal=True,
            help="Select single file or batch process folder"
        )
        
        input_path = None
        
        if input_method == "File Path":
            input_path = st.text_input(
                "Enter CSV file path:",
                placeholder="/path/to/tasks.csv",
                help="Path to CSV file containing tasks"
            )
        else:  # Folder Path
            input_path = st.text_input(
                "Enter folder path:",
                placeholder="/path/to/tasks/folder",
                help="Path to folder containing CSV task files"
            )
        
        st.subheader("Sampling Parameters")
        
        # Sampling method selection
        sampling_method = st.radio(
            "Sampling method:",
            ["Coverage Percentage", "Fixed Count"],
            horizontal=True,
            help="Choose how to sample tasks"
        )
        
        # Multiple files option
        generate_multiple_files = st.checkbox(
            "Generate multiple CSV files without overlap",
            value=False,
            help="Create multiple sampling files with no overlapping tasks"
        )
        
        if generate_multiple_files:
            num_files = st.number_input(
                "Number of files to generate:",
                min_value=2,
                max_value=10,
                value=3,
                help="Number of separate CSV files to create"
            )
            
            st.info("""
            **How multiple files work:**
            - Each file will contain different tasks (no overlap)
            - Tasks are selected using different random seeds
            - Useful for creating separate evaluation sets
            - Each file maintains the same sampling parameters
            """)
        else:
            num_files = 1
        
        if sampling_method == "Coverage Percentage":
            coverage_percentage = st.slider(
                "Coverage percentage per model:",
                min_value=0.01,
                max_value=1.0,
                value=0.3,
                step=0.01,
                format="%.2f",
                help="Percentage of tasks to sample from each model (0.0 = 0%, 1.0 = 100%)"
            )
            
            # Minimum tasks option
            use_minimum_tasks = st.checkbox(
                "Set minimum number of tasks",
                value=False,
                help="Ensure at least a minimum number of tasks are selected, even if percentage would result in fewer"
            )
            
            if use_minimum_tasks:
                minimum_tasks = st.number_input(
                    "Minimum number of tasks per model:",
                    min_value=1,
                    max_value=1000,
                    value=10,
                    help="Minimum number of tasks to select per model (will override percentage if needed)"
                )
            else:
                minimum_tasks = None
            
            if generate_multiple_files:
                info_text = f"""
                **Multiple Files Mode:**
                - Each file will contain {coverage_percentage:.1%} of the original tasks
                - Total unique tasks across all files: ≤ {coverage_percentage * num_files:.1%} of original
                - Files have non-overlapping tasks when possible"""
                
                if use_minimum_tasks:
                    info_text += f"\n                - Minimum {minimum_tasks} tasks per model per file (if available)"
                
                st.info(info_text)
            else:
                info_text = f"ℹ️ Will select {coverage_percentage:.1%} of tasks from the input"
                
                if use_minimum_tasks:
                    info_text += f" (minimum {minimum_tasks} tasks per model if available)"
                
                st.info(info_text)
            
            tasks_count = None
        else:  # Fixed Count
            if task_type == "Word Intrusion":
                # Determine the appropriate label based on input method
                if input_method == "Folder Path":
                    if generate_multiple_files:
                        label_text = "Number of tasks per input file (selected in each output file):"
                        help_text = "Fixed number of tasks to sample from each input file in the folder - this number will be selected in each individual output file"
                    else:
                        label_text = "Number of tasks per input file:"
                        help_text = "Fixed number of tasks to sample from each input file in the folder"
                else:  # File Path
                    if generate_multiple_files:
                        label_text = "Number of tasks per model (selected in each output file):"
                        help_text = "Fixed number of tasks to sample from each model - this number will be selected in each individual output file"
                    else:
                        label_text = "Number of tasks per model:"
                        help_text = "Fixed number of tasks to sample from each model"
                
                tasks_count = st.number_input(
                    label_text,
                    min_value=1,
                    value=100 if not generate_multiple_files else (100 // num_files),
                    help=help_text
                )
                
                if generate_multiple_files:
                    if input_method == "Folder Path":
                        total_tasks_per_input_file = tasks_count * num_files
                        st.info(f"ℹ️ Total tasks per input file across all {num_files} output files: {total_tasks_per_input_file}")
                    else:  # File Path
                        total_tasks_per_model = tasks_count * num_files
                        st.info(f"ℹ️ Total tasks per model across all {num_files} output files: {total_tasks_per_model}")
            else:  # Topic Mixing
                # Determine the appropriate label based on input method
                if input_method == "Folder Path":
                    if generate_multiple_files:
                        label_text = "Total number of tasks (selected in each output file):"
                        help_text = "Total number of tasks to sample from all input files in the folder - this number will be selected in each individual output file (maintaining proportions)"
                    else:
                        label_text = "Total number of tasks:"
                        help_text = "Total number of tasks to sample from all input files in the folder (maintaining proportions)"
                else:  # File Path
                    if generate_multiple_files:
                        label_text = "Total number of tasks (selected in each output file):"
                        help_text = "Total number of tasks to sample - this number will be selected in each individual output file (maintaining proportions)"
                    else:
                        label_text = "Total number of tasks:"
                        help_text = "Total number of tasks to sample (maintaining proportions)"
                
                tasks_count = st.number_input(
                    label_text,
                    min_value=1,
                    value=500 if not generate_multiple_files else (500 // num_files),
                    help=help_text
                )
                
                if generate_multiple_files:
                    total_tasks_all_files = tasks_count * num_files
                    st.info(f"ℹ️ Total tasks across all {num_files} output files: {total_tasks_all_files}")
            
            coverage_percentage = None
            minimum_tasks = None  # Not used in Fixed Count mode
        
        # Random seed
        random_seed = st.number_input(
            "Random seed:",
            min_value=0,
            value=42,
            help="Seed for reproducible sampling"
        )
        
        # Control tasks option
        st.subheader("Control Tasks")
        
        include_control_tasks = st.checkbox(
            "Include control tasks",
            value=False,
            help="Add control tasks to detect annotator attention"
        )
        
        control_tasks_file = None
        control_tasks_per_file = 0
        mixing_control_single_count = 0
        mixing_control_two_count = 0
        
        if include_control_tasks:
            # For Topic Mixing, default to the specific mixing control tasks file
            if task_type == "Topic Mixing":
                control_tasks_file = st.text_input(
                    "Control tasks CSV file:",
                    value="/home/tproutea/mixing_control_tasks.csv",
                    help="CSV file containing topic mixing control tasks with type markers (Q=-10 for single-topic, Q=-20 for two-topic)"
                )
            else:
                control_tasks_file = st.text_input(
                    "Control tasks CSV file:",
                    placeholder="/path/to/control_tasks.csv",
                    help="CSV file containing control tasks with known correct answers"
                )
            
            if control_tasks_file and Path(control_tasks_file).exists():
                try:
                    control_df = pd.read_csv(control_tasks_file)
                    
                    if task_type == "Topic Mixing":
                        # Analyze mixing control tasks by quartile
                        quartile_counts = control_df['quartile'].value_counts().sort_index()
                        single_topic_tasks = len(control_df[control_df['quartile'] == -10])
                        two_topic_tasks = len(control_df[control_df['quartile'] == -20])
                        
                        st.success(f"✅ Found {len(control_df)} mixing control tasks")
                        
                        col_ctrl1, col_ctrl2 = st.columns(2)
                        with col_ctrl1:
                            st.metric("Single-topic (Q=-10)", single_topic_tasks)
                        with col_ctrl2:
                            st.metric("Two-topic (Q=-20)", two_topic_tasks)
                        
                        # Control task quantity selection for mixing
                        st.write("**Control Task Selection:**")
                        col_mixing1, col_mixing2 = st.columns(2)
                        
                        with col_mixing1:
                            mixing_control_single_count = st.number_input(
                                "Single-topic control tasks:",
                                min_value=0,
                                max_value=single_topic_tasks,
                                value=min(2, single_topic_tasks),
                                help=f"Number of single-topic control tasks (max {single_topic_tasks} available)"
                            )
                        
                        with col_mixing2:
                            mixing_control_two_count = st.number_input(
                                "Two-topic control tasks:",
                                min_value=0,
                                max_value=two_topic_tasks,
                                value=min(2, two_topic_tasks),
                                help=f"Number of two-topic control tasks (max {two_topic_tasks} available)"
                            )
                        
                        control_tasks_per_file = mixing_control_single_count + mixing_control_two_count
                        
                        if control_tasks_per_file > 0:
                            st.info(f"""
                            **Mixing Control Tasks Info:**
                            - {mixing_control_single_count} single-topic + {mixing_control_two_count} two-topic = {control_tasks_per_file} total control tasks per file
                            - Equal representation ensures balanced evaluation
                            - Control tasks help detect if annotators can distinguish coherent vs mixed topics
                            {"- Each file gets the SAME control tasks for consistency" if generate_multiple_files else ""}
                            """)
                        
                    else:
                        # Standard control tasks handling for Word Intrusion
                        st.success(f"✅ Found {len(control_df)} control tasks")
                        
                        control_tasks_per_file = st.number_input(
                            f"Control tasks per file{' (each file will get the same control tasks)' if generate_multiple_files else ''}:",
                            min_value=0,
                            max_value=len(control_df),
                            value=min(5, len(control_df)),
                            help="Number of control tasks to include in each output file"
                        )
                        
                        if control_tasks_per_file > 0:
                            st.info(f"""
                            **Control Tasks Info:**
                            - {control_tasks_per_file} control tasks will be added to each output file
                            - Control tasks help detect if annotators are paying attention
                            - They will be randomly distributed among the sampled tasks
                            {"- Each file gets the SAME control tasks for consistency" if generate_multiple_files else ""}
                            """)
                    
                    # Show a preview of control tasks structure
                    with st.expander("Preview control tasks structure"):
                        st.dataframe(control_df.head(3), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error reading control tasks file: {e}")
            elif control_tasks_file:
                st.error("❌ Control tasks file not found")
        
        # Output settings
        st.subheader("Output Settings")
        
        # Initialize variables to ensure they exist
        output_path = None
        output_dir = None
        file_prefix = None
        
        if generate_multiple_files:
            output_dir = st.text_input(
                "Output directory:",
                value="/home/tproutea/sampled_tasks_batch/",
                help="Directory where multiple CSV files will be saved"
            )
            
            file_prefix = st.text_input(
                "File prefix:",
                value="sampled_tasks",
                help="Prefix for the generated files (e.g., 'sampled_tasks' → 'sampled_tasks_1.csv', 'sampled_tasks_2.csv', ...)"
            )
        else:
            output_path = st.text_input(
                "Output file path:",
                value="/home/tproutea/sampled_tasks.csv",
                help="Path where sampled tasks will be saved"
            )
        
    with col2:
        st.subheader("Sampling Info")
        
        if input_path and Path(input_path).exists():
            try:
                if input_method == "File Path":
                    # Single file
                    df = pd.read_csv(input_path)
                    st.metric("Total Tasks", len(df))
                    
                    # Show model distribution
                    if 'model' in df.columns:
                        model_counts = df['model'].value_counts()
                        st.write("**Tasks per model:**")
                        for model, count in model_counts.items():
                            st.write(f"• {model}: {count}")
                    
                    # Show quartile distribution for mixing tasks
                    if task_type == "Topic Mixing" and 'quartile' in df.columns:
                        quartile_counts = df['quartile'].value_counts()
                        st.write("**Tasks per quartile:**")
                        for quartile in sorted(quartile_counts.index):
                            st.write(f"• Q{quartile}: {quartile_counts[quartile]}")
                
                else:  # Folder path
                    folder = Path(input_path)
                    csv_files = list(folder.glob("*.csv"))
                    st.metric("CSV Files Found", len(csv_files))
                    
                    if csv_files:
                        total_tasks = 0
                        for csv_file in csv_files[:5]:  # Show first 5 files
                            try:
                                df = pd.read_csv(csv_file)
                                total_tasks += len(df)
                                st.write(f"• {csv_file.name}: {len(df)} tasks")
                            except:
                                continue
                        
                        if len(csv_files) > 5:
                            st.write(f"... and {len(csv_files) - 5} more files")
                        
                        st.metric("Total Tasks (estimated)", total_tasks)
                        
            except Exception as e:
                st.error(f"Error reading input: {e}")
        
        # Control tasks info
        st.subheader("ℹ️ About Control Tasks")
        
        if task_type == "Topic Mixing":
            st.markdown("""
            **Mixing control tasks** help validate annotation quality:
            
            - **Single-topic (Q=-10)**: Words from same coherent topic
            - **Two-topic (Q=-20)**: Words from clearly different topics  
            - **Purpose**: Test if annotators can distinguish topic coherence
            - **Balance**: Equal representation of both control types
            - **Distribution**: Randomly mixed with main tasks
            - **Consistency**: Same control tasks in all files
            """)
        else:
            st.markdown("""
            **Control tasks** help detect if annotators are paying attention:
            
            - **Purpose**: Tasks with known correct answers
            - **Detection**: Identify inattentive annotators
            - **Distribution**: Randomly mixed with main tasks
            - **Consistency**: Same control tasks in all files
            - **Format**: CSV with task structure matching main tasks
            """)
        
        if include_control_tasks:
            if task_type == "Topic Mixing" and (mixing_control_single_count > 0 or mixing_control_two_count > 0):
                st.success(f"✅ Mixing control tasks enabled: {mixing_control_single_count} single + {mixing_control_two_count} two-topic")
            elif task_type != "Topic Mixing" and control_tasks_per_file > 0:
                st.success(f"✅ Control tasks enabled: {control_tasks_per_file} tasks")
            else:
                st.success("✅ Control tasks enabled")
        else:
            st.info("💡 Enable control tasks for quality control")
        
        # Preview sampling button
        if st.button("🔍 Preview Sampling", type="primary"):
            if not input_path:
                st.error("Please provide an input path")
            elif not Path(input_path).exists():
                st.error("Input path does not exist")
            elif generate_multiple_files and not output_dir:
                st.error("Please provide an output directory for multiple files")
            elif not generate_multiple_files and not output_path:
                st.error("Please provide an output file path")
            elif include_control_tasks and not control_tasks_file:
                st.error("Please provide control tasks file path")
            elif include_control_tasks and control_tasks_file and not Path(control_tasks_file).exists():
                st.error("Control tasks file does not exist")
            else:
                with st.spinner("Generating preview..."):
                    try:
                        # Load control tasks if specified
                        control_tasks = None
                        if include_control_tasks and control_tasks_file and control_tasks_per_file > 0:
                            control_df = pd.read_csv(control_tasks_file)
                            
                            if task_type == "Topic Mixing":
                                # Handle mixing control tasks with quartile-based selection
                                control_tasks = []
                                
                                # Select single-topic control tasks (quartile = -10)
                                if mixing_control_single_count > 0:
                                    single_topic_tasks = control_df[control_df['quartile'] == -10]
                                    if len(single_topic_tasks) >= mixing_control_single_count:
                                        selected_single = single_topic_tasks.sample(n=mixing_control_single_count, random_state=random_seed)
                                        control_tasks.extend(selected_single.to_dict('records'))
                                    else:
                                        st.warning(f"Only {len(single_topic_tasks)} single-topic control tasks available, but {mixing_control_single_count} requested. Using all available.")
                                        control_tasks.extend(single_topic_tasks.to_dict('records'))
                                
                                # Select two-topic control tasks (quartile = -20)
                                if mixing_control_two_count > 0:
                                    two_topic_tasks = control_df[control_df['quartile'] == -20]
                                    if len(two_topic_tasks) >= mixing_control_two_count:
                                        selected_two = two_topic_tasks.sample(n=mixing_control_two_count, random_state=random_seed + 1)  # Different seed for variety
                                        control_tasks.extend(selected_two.to_dict('records'))
                                    else:
                                        st.warning(f"Only {len(two_topic_tasks)} two-topic control tasks available, but {mixing_control_two_count} requested. Using all available.")
                                        control_tasks.extend(two_topic_tasks.to_dict('records'))
                                
                            else:
                                # Standard control task handling for Word Intrusion
                                if len(control_df) >= control_tasks_per_file:
                                    control_tasks = control_df.sample(n=control_tasks_per_file, random_state=random_seed).to_dict('records')
                                else:
                                    st.warning(f"Control file has only {len(control_df)} tasks, but {control_tasks_per_file} requested. Using all available.")
                                    control_tasks = control_df.to_dict('records')
                            
                            # Mark control tasks for identification
                            if control_tasks:
                                for control_task in control_tasks:
                                    control_task['is_control_task'] = True
                        
                        # Perform sampling based on method
                        selected = []  # Initialize selected tasks list
                        if input_method == "File Path":
                            # Single file sampling
                            if task_type == "Word Intrusion":
                                tasks = TaskSelector.load_word_intrusion_tasks(input_path)
                                # Store for debugging
                                st.session_state.all_tasks_debug_count = len(tasks)
                                if coverage_percentage is not None:
                                    if generate_multiple_files:
                                        # Preview first file only - use same logic as save to ensure consistency
                                        selected = TaskSelector.select_word_intrusion_tasks(
                                            tasks, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                                        st.info(f"Preview shows what File 1 of {num_files} will contain. Other files will have different tasks (no overlap).")
                                    else:
                                        selected = TaskSelector.select_word_intrusion_tasks(
                                            tasks, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                                else:
                                    if generate_multiple_files:
                                        # Preview first file only - use same logic as save to ensure consistency
                                        selected = TaskSelector.select_word_intrusion_tasks(
                                            tasks, tasks_per_model=tasks_count, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                                        st.info(f"Preview shows what File 1 of {num_files} will contain. Other files will have different tasks (no overlap).")
                                    else:
                                        selected = TaskSelector.select_word_intrusion_tasks(
                                            tasks, tasks_per_model=tasks_count, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                            else:  # Topic Mixing
                                tasks = TaskSelector.load_mixing_tasks(input_path)
                                # Store for debugging
                                st.session_state.all_tasks_debug_count = len(tasks)
                                if coverage_percentage is not None:
                                    if generate_multiple_files:
                                        # Preview first file only - use same logic as save to ensure consistency
                                        selected = TaskSelector.select_mixing_tasks(
                                            tasks, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                                        st.info(f"Preview shows what File 1 of {num_files} will contain. Other files will have different tasks (no overlap).")
                                    else:
                                        selected = TaskSelector.select_mixing_tasks(
                                            tasks, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                                else:
                                    if generate_multiple_files:
                                        # Preview first file only - use same logic as save to ensure consistency
                                        selected = TaskSelector.select_mixing_tasks(
                                            tasks, total_tasks=tasks_count, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                                        st.info(f"Preview shows what File 1 of {num_files} will contain. Other files will have different tasks (no overlap).")
                                    else:
                                        selected = TaskSelector.select_mixing_tasks(
                                            tasks, total_tasks=tasks_count, random_seed=random_seed
                                        )
                                        st.session_state.sampling_preview = selected
                        else:
                            # Folder sampling
                            if task_type == "Word Intrusion":
                                if coverage_percentage is not None:
                                    results = process_word_intrusion_folder(
                                        input_path, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=random_seed
                                    )
                                else:
                                    results = process_word_intrusion_folder(
                                        input_path, tasks_per_model=tasks_count, random_seed=random_seed
                                    )
                            else:  # Topic Mixing
                                if coverage_percentage is not None:
                                    results = process_mixing_folder(
                                        input_path, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=random_seed
                                    )
                                else:
                                    results = process_mixing_folder(
                                        input_path, total_tasks=tasks_count, random_seed=random_seed
                                    )
                            
                            # Combine all results
                            selected = []
                            for file_tasks in results.values():
                                selected.extend(file_tasks)
                            
                            if generate_multiple_files:
                                st.info(f"Folder preview shows combined results. Multiple files will be generated without overlap.")
                        
                        # Add control tasks to preview if specified
                        if control_tasks:
                            # Add control tasks to selected tasks for preview
                            selected_with_controls = selected + control_tasks
                            
                            # Shuffle to distribute control tasks randomly
                            import random
                            random.seed(random_seed)
                            random.shuffle(selected_with_controls)
                            
                            st.session_state.sampling_preview = selected_with_controls
                                
                            st.info(f"Added {len(control_tasks)} control tasks. Total preview: {len(selected_with_controls)} tasks")
                        else:
                            st.session_state.sampling_preview = selected
                        
                        st.session_state.sampling_results = {
                            'input_path': input_path,
                            'input_method': input_method,
                            'task_type': task_type,
                            'sampling_method': sampling_method,
                            'coverage_percentage': coverage_percentage,
                            'tasks_count': tasks_count,
                            'minimum_tasks': minimum_tasks,
                            'random_seed': random_seed,
                            'generate_multiple_files': generate_multiple_files,
                            'num_files': num_files,
                            'output_path': output_path,
                            'output_dir': output_dir,
                            'file_prefix': file_prefix,
                            'include_control_tasks': include_control_tasks,
                            'control_tasks_file': control_tasks_file,
                            'control_tasks_per_file': control_tasks_per_file,
                            'control_tasks': control_tasks,
                            'mixing_control_single_count': mixing_control_single_count if task_type == "Topic Mixing" else 0,
                            'mixing_control_two_count': mixing_control_two_count if task_type == "Topic Mixing" else 0
                        }
                        
                        if control_tasks:
                            st.success(f"Preview generated! {len(selected)} main tasks + {len(control_tasks)} control tasks = {len(selected) + len(control_tasks)} total tasks.")
                        else:
                            st.success(f"Preview generated! {len(selected)} tasks selected.")
                        
                        # Debug info for multiple files
                        if results.get('generate_multiple_files', False):
                            st.info(f"Preview: Using seed {random_seed}, sampled {len(selected)} main tasks from {st.session_state.get('all_tasks_debug_count', 'unknown')} total tasks")
                        
                    except Exception as e:
                        st.error(f"Error during sampling: {e}")
                        st.text(traceback.format_exc())
    
    # Preview section
    if st.session_state.sampling_preview is not None:
        st.subheader("📋 Sampling Preview")
        
        selected_tasks = st.session_state.sampling_preview
        results = st.session_state.sampling_results
        
        # Add info about multiple files
        if results.get('generate_multiple_files', False):
            st.info(f"""
            **Multiple Files Mode**: {results['num_files']} files will be generated
            - Preview shows what ONE file will look like
            - Each file will have similar content but NO overlapping tasks
            - Total tasks across all files: ~{len(selected_tasks) * results['num_files']}
            """)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Tasks", len(selected_tasks))
        
        # Check for control tasks
        control_tasks_count = len([task for task in selected_tasks if task.get('is_control_task', False)])
        main_tasks_count = len(selected_tasks) - control_tasks_count
        
        if control_tasks_count > 0:
            with col2:
                st.metric("Main Tasks", main_tasks_count)
            with col3:
                st.metric("Control Tasks", control_tasks_count)
        
        if selected_tasks:
            # Control tasks information
            if control_tasks_count > 0:
                if task_type == "Topic Mixing":
                    # Show control task breakdown by quartile for mixing
                    control_tasks = [task for task in selected_tasks if task.get('is_control_task', False)]
                    control_quartiles = [task.get('quartile', 'unknown') for task in control_tasks]
                    control_quartile_counts = pd.Series(control_quartiles).value_counts()
                    
                    control_single = control_quartile_counts.get(-10, 0)
                    control_two = control_quartile_counts.get(-20, 0)
                    
                    st.info(f"📋 Control tasks: {control_single} single-topic (Q=-10) + {control_two} two-topic (Q=-20) = {control_tasks_count} total")
                else:
                    st.info(f"📋 This preview includes {control_tasks_count} control tasks mixed with {main_tasks_count} main tasks")
            
            # Model distribution (only for main tasks)
            main_tasks = [task for task in selected_tasks if not task.get('is_control_task', False)]
            if main_tasks and any('model' in task for task in main_tasks):
                models = [task.get('model', 'unknown') for task in main_tasks]
                model_counts = pd.Series(models).value_counts()
                
                # Show distribution
                st.write("**Main tasks per model:**")
                for model, count in model_counts.items():
                    st.write(f"• {model}: {count} tasks")
            
            # Quartile distribution for mixing tasks (only main tasks)
            if task_type == "Topic Mixing" and main_tasks and any('quartile' in task for task in main_tasks):
                quartiles = [task.get('quartile', 'unknown') for task in main_tasks]
                quartile_counts = pd.Series(quartiles).value_counts().sort_index()
                
                # Separate single-topic from mixed-topic
                single_topic_count = quartile_counts.get(-1, 0)
                mixed_quartiles = {q: count for q, count in quartile_counts.items() if isinstance(q, (int, float)) and q >= 0}
                mixed_total = sum(mixed_quartiles.values())
                
                st.write("**Main tasks per quartile (Sampling Strategy Applied):**")
                
                # Show single-topic tasks
                if single_topic_count > 0:
                    percentage = (single_topic_count / len(main_tasks)) * 100
                    st.write(f"• **Single-topic (Q=-1)**: {single_topic_count} tasks ({percentage:.1f}%) - Prioritized")
                
                # Show mixed-topic tasks
                if mixed_quartiles:
                    st.write(f"• **Mixed-topic quartiles (Q≥0)**: {mixed_total} tasks total - Equal distribution:")
                    for quartile in sorted(mixed_quartiles.keys()):
                        count = mixed_quartiles[quartile]
                        st.write(f"  - Q{quartile}: {count} tasks")
                
                # Verify the strategy is working
                if single_topic_count > 0 and mixed_total > 0:
                    ratio = single_topic_count / mixed_total
                    expected_ratio = 0.6 / 0.4  # 1.5
                    if abs(ratio - expected_ratio) < 0.3:  # Allow some tolerance
                        st.success(f"✅ Sampling strategy applied: Single-topic to mixed-topic ratio = {ratio:.2f} (target: 1.5)")
                    else:
                        st.info(f"ℹ️ Actual ratio: {ratio:.2f} (target: 1.5) - Limited by available tasks")
            
            # Show sample data
            st.write("**Sample of selected tasks:**")
            df_preview = pd.DataFrame(selected_tasks[:10])  # Show first 10
            
            # Highlight control tasks in the preview
            if control_tasks_count > 0:
                # Add visual indicator for control tasks
                def highlight_control_tasks(row):
                    if row.get('is_control_task', False):
                        return ['background-color: #fff3cd'] * len(row)
                    return [''] * len(row)
                
                styled_df = df_preview.style.apply(highlight_control_tasks, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                st.caption("🟡 Yellow rows are control tasks")
            else:
                st.dataframe(df_preview, use_container_width=True)
            
            if len(selected_tasks) > 10:
                st.write(f"... and {len(selected_tasks) - 10} more tasks")
            
            # Save button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("💾 Save Selected Tasks", type="primary"):
                    try:
                        results = st.session_state.sampling_results
                        
                        if results.get('generate_multiple_files', False):
                            # Generate multiple files without overlap
                            st.info("Generating multiple files with no overlapping tasks...")
                            save_multiple_sampling_files(results, selected_tasks)
                        else:
                            # Single file save
                            output_file = Path(results['output_path'])
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            df_save = pd.DataFrame(selected_tasks)
                            df_save.to_csv(output_file, index=False)
                            
                            # Show save details
                            main_tasks_count = len([task for task in selected_tasks if not task.get('is_control_task', False)])
                            control_tasks_count = len([task for task in selected_tasks if task.get('is_control_task', False)])
                            
                            if control_tasks_count > 0:
                                st.success(f"✅ Saved {main_tasks_count} main tasks + {control_tasks_count} control tasks = {len(selected_tasks)} total tasks to {output_file}")
                            else:
                                st.success(f"✅ Saved {len(selected_tasks)} tasks to {output_file}")
                        
                    except Exception as e:
                        st.error(f"Error saving file: {e}")
            
            with col2:
                # Download options
                if not st.session_state.sampling_results.get('generate_multiple_files', False):
                    # Single file download - uses exact preview tasks
                    df_download = pd.DataFrame(selected_tasks)
                    csv_data = df_download.to_csv(index=False)
                    st.download_button(
                        "📥 Download Preview Tasks",
                        data=csv_data,
                        file_name=f"sampled_{task_type.lower().replace(' ', '_')}_tasks.csv",
                        mime="text/csv",
                        help="Download the exact tasks shown in preview",
                        type="primary"
                    )
                else:
                    # Multiple files - offer different options
                    st.subheader("📥 Download Options")
                    
                    # Option 1: Download preview as single file
                    st.write("**Option 1: Single File (Exact Preview)**")
                    df_download = pd.DataFrame(selected_tasks)
                    csv_data = df_download.to_csv(index=False)
                    st.download_button(
                        "� Download as Single File",
                        data=csv_data,
                        file_name=f"preview_{task_type.lower().replace(' ', '_')}_tasks.csv",
                        mime="text/csv",
                        help="Download exactly what you see in preview as one file"
                    )
                    
                    # Option 2: Download multiple files from preview distribution
                    st.write("**Option 2: Multiple Files (From Preview)**")
                    if st.button("📦 Generate Multiple Files from Preview", help="Split preview tasks into multiple files", type="primary"):
                        download_multiple_files_from_preview(selected_tasks, st.session_state.sampling_results)
                    
                    # Option 3: Regenerate multiple files
                    st.write("**Option 3: Regenerated Multiple Files**")
                    st.info("💡 Use 'Save Selected Tasks' to regenerate tasks with no overlap (may differ from preview)")


def download_multiple_files_from_preview(selected_tasks, results):
    """Generate multiple files for download based on the preview tasks"""
    try:
        import zipfile
        import io
        from datetime import datetime
        
        # Get parameters
        num_files = results['num_files']
        file_prefix = results.get('file_prefix', 'sampled_tasks')
        task_type = results['task_type']
        
        # Separate main tasks and control tasks
        main_tasks = [task for task in selected_tasks if not task.get('is_control_task', False)]
        control_tasks = [task for task in selected_tasks if task.get('is_control_task', False)]
        
        # Check if we have enough main tasks for distribution
        if len(main_tasks) < num_files:
            st.error(f"Not enough main tasks ({len(main_tasks)}) to distribute across {num_files} files. Need at least {num_files} main tasks.")
            return
        
        # Calculate tasks per file
        tasks_per_file = len(main_tasks) // num_files
        remainder = len(main_tasks) % num_files
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            start_idx = 0
            
            for file_idx in range(num_files):
                # Calculate how many tasks for this file (distribute remainder)
                current_file_tasks = tasks_per_file + (1 if file_idx < remainder else 0)
                end_idx = start_idx + current_file_tasks
                
                # Get main tasks for this file
                file_main_tasks = main_tasks[start_idx:end_idx]
                
                # Add control tasks to each file
                file_tasks = file_main_tasks.copy()
                file_tasks.extend(control_tasks)  # Control tasks go in every file
                
                # Shuffle tasks to distribute control tasks randomly
                import random
                random.seed(results['random_seed'] + file_idx)
                random.shuffle(file_tasks)
                
                # Create filename
                filename = f"{file_prefix}_{file_idx + 1}_{timestamp}.csv"
                
                # Convert to CSV
                df = pd.DataFrame(file_tasks)
                csv_content = df.to_csv(index=False)
                
                # Add to zip
                zip_file.writestr(filename, csv_content)
                
                start_idx = end_idx
        
        zip_buffer.seek(0)
        
        # Offer download
        st.download_button(
            "📦 Download ZIP with Multiple Files",
            data=zip_buffer.read(),
            file_name=f"{file_prefix}_multiple_{timestamp}.zip",
            mime="application/zip",
            help=f"Downloads a ZIP containing {num_files} CSV files with distributed preview tasks"
        )
        
        # Show distribution details
        st.success(f"✅ Created {num_files} files from {len(main_tasks)} main tasks + {len(control_tasks)} control tasks")
        
        # Show file details
        start_idx = 0
        for file_idx in range(num_files):
            current_file_tasks = tasks_per_file + (1 if file_idx < remainder else 0)
            total_with_controls = current_file_tasks + len(control_tasks)
            st.write(f"• File {file_idx + 1}: {current_file_tasks} main + {len(control_tasks)} control = {total_with_controls} total tasks")
            start_idx += current_file_tasks
        
        if len(control_tasks) > 0:
            st.info("ℹ️ Control tasks are included in all files for consistency")
        
    except Exception as e:
        st.error(f"Error generating multiple files: {e}")
        import traceback
        st.text(traceback.format_exc())


def save_multiple_sampling_files(results, selected_tasks):
    """Save multiple sampling files without overlap between tasks"""
    try:
        import numpy as np
        from datetime import datetime
        
        # Create output directory
        output_dir = Path(results['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_prefix = results.get('file_prefix', 'sampled_tasks')
        num_files = results['num_files']
        task_type = results['task_type']
        input_method = results['input_method']
        input_path = results['input_path']
        sampling_method = results['sampling_method']
        coverage_percentage = results['coverage_percentage']
        tasks_count = results['tasks_count']
        minimum_tasks = results.get('minimum_tasks', None)
        random_seed = results['random_seed']
        
        # Handle control tasks
        include_control_tasks = results.get('include_control_tasks', False)
        control_tasks = results.get('control_tasks', [])
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Re-generate tasks for multiple files with different seeds to avoid overlap
        all_files_data = []
        
        if input_method == "File Path":
            # Single file - generate multiple non-overlapping samples
            if task_type == "Word Intrusion":
                all_tasks = TaskSelector.load_word_intrusion_tasks(input_path)
            else:  # Topic Mixing
                all_tasks = TaskSelector.load_mixing_tasks(input_path)
            
            # Calculate total tasks needed (excluding control tasks from main sampling)
            if sampling_method == "Coverage Percentage":
                total_needed = int(len(all_tasks) * coverage_percentage * num_files)
            else:  # Fixed count
                total_needed = tasks_count * num_files
            
            # Check if we have enough tasks
            if total_needed > len(all_tasks):
                st.warning(f"Not enough tasks available. Requested: {total_needed}, Available: {len(all_tasks)}")
                total_needed = len(all_tasks)
            
            # Generate non-overlapping samples
            used_task_ids = set()
            
            for file_idx in range(num_files):
                # Get available tasks (exclude already used ones)
                available_tasks = [task for task in all_tasks if task.get('text', task.get('task_id', f"task_{hash(str(task))}")) not in used_task_ids]
                
                if not available_tasks:
                    st.warning(f"No more tasks available for file {file_idx + 1}")
                    break
                
                # Sample from available tasks
                current_seed = random_seed + file_idx * 1000
                
                # Debug info for first file to verify consistency with preview
                if file_idx == 0:
                    st.info(f"File 1: Using seed {current_seed}, sampling from {len(available_tasks)} available tasks")
                
                if task_type == "Word Intrusion":
                    if sampling_method == "Coverage Percentage":
                        # Adjust coverage to available tasks
                        adjusted_coverage = min(coverage_percentage, len(available_tasks) / len(all_tasks))
                        file_tasks = TaskSelector.select_word_intrusion_tasks(
                            available_tasks, coverage_percentage=adjusted_coverage, minimum_tasks=minimum_tasks, random_seed=current_seed
                        )
                    else:
                        file_tasks = TaskSelector.select_word_intrusion_tasks(
                            available_tasks, tasks_per_model=min(tasks_count, len(available_tasks)), random_seed=current_seed
                        )
                else:  # Topic Mixing
                    if sampling_method == "Coverage Percentage":
                        adjusted_coverage = min(coverage_percentage, len(available_tasks) / len(all_tasks))
                        file_tasks = TaskSelector.select_mixing_tasks(
                            available_tasks, coverage_percentage=adjusted_coverage, minimum_tasks=minimum_tasks, random_seed=current_seed
                        )
                    else:
                        file_tasks = TaskSelector.select_mixing_tasks(
                            available_tasks, total_tasks=min(tasks_count, len(available_tasks)), random_seed=current_seed
                        )
                
                # Add control tasks to each file
                tasks_with_controls = file_tasks.copy()
                if include_control_tasks and control_tasks:
                    # Add control tasks to this file
                    for control_task in control_tasks:
                        control_task_copy = control_task.copy()
                        control_task_copy['is_control_task'] = True
                        tasks_with_controls.append(control_task_copy)
                    
                    # Shuffle to distribute control tasks randomly
                    import random
                    random.seed(current_seed)
                    random.shuffle(tasks_with_controls)
                
                # Mark used task IDs
                for task in file_tasks:
                    task_id = task.get('text', task.get('task_id', f"task_{hash(str(task))}"))
                    used_task_ids.add(task_id)
                
                # Save file
                filename = f"{file_prefix}_{file_idx + 1}_{timestamp}.csv"
                file_path = output_dir / filename
                
                df = pd.DataFrame(tasks_with_controls)
                df.to_csv(file_path, index=False)
                
                all_files_data.append({
                    'filename': filename,
                    'tasks_count': len(tasks_with_controls),
                    'main_tasks_count': len(file_tasks),
                    'control_tasks_count': len(control_tasks) if include_control_tasks else 0,
                    'tasks': tasks_with_controls
                })
        
        else:  # Folder method
            # For folder processing, we need to handle differently
            # This is more complex as we need to distribute across models/files
            folder_path = Path(input_path)
            
            # Load all tasks from folder using the process_folder method
            # This returns a dict with filename -> tasks mapping
            if task_type == "Word Intrusion":
                folder_results = TaskSelector.process_folder(
                    folder_path=folder_path,
                    task_type='word_intrusion',
                    coverage_percentage=1.0,  # Load all tasks first
                    random_seed=random_seed
                )
            else:  # Topic Mixing
                folder_results = TaskSelector.process_folder(
                    folder_path=folder_path,
                    task_type='mixing',
                    coverage_percentage=1.0,  # Load all tasks first
                    random_seed=random_seed
                )
            
            # Flatten all tasks into a single list
            all_tasks = []
            for file_tasks in folder_results.values():
                all_tasks.extend(file_tasks)
            
            # Calculate total tasks needed per file
            if sampling_method == "Coverage Percentage":
                tasks_per_file = int(len(all_tasks) * coverage_percentage)
            else:  # Fixed count
                tasks_per_file = tasks_count
            
            # Check if we have enough tasks
            total_needed = tasks_per_file * num_files
            if total_needed > len(all_tasks):
                st.warning(f"Not enough tasks available. Requested: {total_needed}, Available: {len(all_tasks)}")
                tasks_per_file = len(all_tasks) // num_files
                if tasks_per_file == 0:
                    st.error("Not enough tasks to create any files")
                    return
            
            # Generate non-overlapping samples for folder input
            used_task_indices = set()
            
            for file_idx in range(num_files):
                # Get available task indices
                available_indices = [i for i in range(len(all_tasks)) if i not in used_task_indices]
                
                if len(available_indices) < tasks_per_file:
                    st.warning(f"Not enough tasks available for file {file_idx + 1}. Using {len(available_indices)} tasks.")
                    if not available_indices:
                        break
                
                # Sample tasks for this file
                current_seed = random_seed + file_idx * 1000
                np.random.seed(current_seed)
                
                sample_size = min(tasks_per_file, len(available_indices))
                selected_indices = np.random.choice(available_indices, size=sample_size, replace=False)
                
                # Get selected tasks
                file_tasks = [all_tasks[i] for i in selected_indices]
                
                # Add control tasks to each file
                tasks_with_controls = file_tasks.copy()
                if include_control_tasks and control_tasks:
                    # Add control tasks to this file
                    for control_task in control_tasks:
                        control_task_copy = control_task.copy()
                        control_task_copy['is_control_task'] = True
                        tasks_with_controls.append(control_task_copy)
                    
                    # Shuffle to distribute control tasks randomly
                    import random
                    random.seed(current_seed)
                    random.shuffle(tasks_with_controls)
                
                # Mark used task indices
                used_task_indices.update(selected_indices)
                
                # Save file
                filename = f"{file_prefix}_{file_idx + 1}_{timestamp}.csv"
                file_path = output_dir / filename
                
                df = pd.DataFrame(tasks_with_controls)
                df.to_csv(file_path, index=False)
                
                all_files_data.append({
                    'filename': filename,
                    'tasks_count': len(tasks_with_controls),
                    'main_tasks_count': len(file_tasks),
                    'control_tasks_count': len(control_tasks) if include_control_tasks else 0,
                    'tasks': tasks_with_controls
                })
        
        # Show success message with details
        st.success(f"✅ Generated {len(all_files_data)} files successfully!")
        
        # Show file details
        st.write("**Generated files:**")
        for i, file_data in enumerate(all_files_data, 1):
            main_count = file_data['main_tasks_count']
            control_count = file_data['control_tasks_count']
            total_count = file_data['tasks_count']
            
            if control_count > 0:
                st.write(f"• File {i}: `{file_data['filename']}` - {main_count} main + {control_count} control = {total_count} total tasks")
            else:
                st.write(f"• File {i}: `{file_data['filename']}` - {total_count} tasks")
        
        total_main_tasks = sum(f['main_tasks_count'] for f in all_files_data)
        total_control_tasks = sum(f['control_tasks_count'] for f in all_files_data)
        total_tasks = sum(f['tasks_count'] for f in all_files_data)
        
        st.info(f"📊 Total across all files: {total_main_tasks} main tasks + {total_control_tasks} control tasks = {total_tasks} total tasks")
        st.info(f"💾 Files saved to: {output_dir}")
        
        # Show overlap verification (only for main tasks)
        main_task_ids = []
        for file_data in all_files_data:
            for task in file_data['tasks']:
                if not task.get('is_control_task', False):
                    task_id = task.get('text', task.get('task_id', f"task_{hash(str(task))}"))
                    main_task_ids.append(task_id)
        
        unique_main_task_ids = set(main_task_ids)
        overlap_count = len(main_task_ids) - len(unique_main_task_ids)
        
        if overlap_count == 0:
            st.success("✅ No overlapping main tasks between files")
        else:
            st.warning(f"⚠️ Found {overlap_count} overlapping main tasks")
        
        if include_control_tasks:
            st.info("ℹ️ Control tasks are intentionally duplicated across all files for consistency")
        
    except Exception as e:
        st.error(f"❌ Error generating multiple files: {e}")
        if st.checkbox("Show detailed error", key="multiple_files_error"):
            st.error(traceback.format_exc())
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Re-generate tasks for multiple files with different seeds to avoid overlap
        all_files_data = []
        
        if input_method == "File Path":
            # Single file - generate multiple non-overlapping samples
            if task_type == "Word Intrusion":
                original_tasks = TaskSelector.load_word_intrusion_tasks(input_path)
            else:  # Topic Mixing
                original_tasks = TaskSelector.load_mixing_tasks(input_path)
            
            # Calculate total tasks needed
            if sampling_method == "Coverage Percentage":
                if task_type == "Word Intrusion":
                    # Calculate how many tasks we need per model per file
                    models = set(task.get('model', 'unknown') for task in original_tasks)
                    tasks_per_model_per_file = {}
                    for model in models:
                        model_tasks = [t for t in original_tasks if t.get('model') == model]
                        tasks_needed = int(len(model_tasks) * coverage_percentage)
                        tasks_per_model_per_file[model] = tasks_needed
                else:  # Topic Mixing
                    tasks_per_file = int(len(original_tasks) * coverage_percentage)
            else:  # Fixed Count
                if task_type == "Word Intrusion":
                    tasks_per_model_per_file = tasks_count
                else:  # Topic Mixing
                    tasks_per_file = tasks_count
            
            # Generate non-overlapping samples
            used_task_ids = set()
            
            for file_idx in range(num_files):
                # Use different seed for each file
                file_seed = random_seed + file_idx * 1000
                
                if task_type == "Word Intrusion":
                    if sampling_method == "Coverage Percentage":
                        # Calculate effective coverage for remaining tasks
                        remaining_tasks = [t for t in original_tasks if t.get('text', '') not in used_task_ids]
                        if not remaining_tasks:
                            st.warning(f"Not enough tasks for file {file_idx + 1}")
                            break
                        
                        selected = TaskSelector.select_word_intrusion_tasks(
                            remaining_tasks, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=file_seed
                        )
                    else:
                        # Fixed count
                        remaining_tasks = [t for t in original_tasks if t.get('text', '') not in used_task_ids]
                        if not remaining_tasks:
                            st.warning(f"Not enough tasks for file {file_idx + 1}")
                            break
                        
                        selected = TaskSelector.select_word_intrusion_tasks(
                            remaining_tasks, tasks_per_model=tasks_count, random_seed=file_seed
                        )
                else:  # Topic Mixing
                    remaining_tasks = [t for t in original_tasks if t.get('task_id', '') not in used_task_ids]
                    if not remaining_tasks:
                        st.warning(f"Not enough tasks for file {file_idx + 1}")
                        break
                    
                    if sampling_method == "Coverage Percentage":
                        selected = TaskSelector.select_mixing_tasks(
                            remaining_tasks, coverage_percentage=coverage_percentage, minimum_tasks=minimum_tasks, random_seed=file_seed
                        )
                    else:
                        selected = TaskSelector.select_mixing_tasks(
                            remaining_tasks, total_tasks=tasks_count, random_seed=file_seed
                        )
                
                # Track used task IDs
                for task in selected:
                    if task_type == "Word Intrusion":
                        used_task_ids.add(task.get('text', ''))
                    else:  # Topic Mixing
                        used_task_ids.add(task.get('task_id', ''))
                
                # Save file
                filename = f"{file_prefix}_{file_idx + 1}_{timestamp}.csv"
                file_path = output_dir / filename
                
                df = pd.DataFrame(selected)
                df.to_csv(file_path, index=False)
                
                all_files_data.append({
                    'filename': filename,
                    'path': file_path,
                    'tasks_count': len(selected),
                    'tasks': selected
                })
        
        else:  # Folder method
            # For folder processing, we need to handle differently
            # This is more complex as we need to distribute across models/files
            st.warning("Multiple file generation for folder input is not yet implemented")
            return
        
        # Show success message with details
        st.success(f"✅ Generated {len(all_files_data)} files successfully!")
        
        # Show file details
        st.write("**Generated files:**")
        for i, file_data in enumerate(all_files_data, 1):
            st.write(f"• File {i}: `{file_data['filename']}` - {file_data['tasks_count']} tasks")
        
        total_tasks = sum(f['tasks_count'] for f in all_files_data)
        st.info(f"📊 Total tasks across all files: {total_tasks}")
        st.info(f"💾 Files saved to: {output_dir}")
        
        # Show overlap verification
        all_task_ids = []
        for file_data in all_files_data:
            for task in file_data['tasks']:
                if task_type == "Word Intrusion":
                    all_task_ids.append(task.get('text', ''))
                else:
                    all_task_ids.append(task.get('task_id', ''))
        
        unique_task_ids = set(all_task_ids)
        overlap_count = len(all_task_ids) - len(unique_task_ids)
        
        if overlap_count == 0:
            st.success("✅ No overlapping tasks between files")
        else:
            st.warning(f"⚠️ Found {overlap_count} overlapping tasks")
        
    except Exception as e:
        st.error(f"❌ Error generating multiple files: {e}")
        if st.checkbox("Show detailed error", key="multiple_files_error"):
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
