# When Numbers Tell Half the Story: Human-Metric Alignment in Topic Model Evaluation

This repository contains the code and dataset for evaluating topic models through human evaluation tasks, specifically designed to assess the alignment between automatic metrics and human judgment.

## Abstract

Topic models are widely used to uncover latent thematic structures in text corpora, yet evaluating their quality remains challenging, particularly in specialized domains. Existing evaluation methods often focus on automatic metrics like topic coherence and diversity, which may not fully align with human judgment. Human evaluation tasks, such as word intrusion, provide valuable insights but are costly and primarily validated on general-domain corpora. 

This paper introduces **Topic Word Mixing (TWM)**, a novel human evaluation task designed to assess inter-topic distinctness by testing whether annotators can distinguish between word sets from a single topic or mixed topics. TWM complements word intrusion's focus on intra-topic coherence and provides a human-grounded counterpart to topic diversity metrics. 

We evaluate six topic models (LDA, NMF, Top2Vec, BERTopic, CFMF, CFMF-emb) using both automatic metrics and human evaluations, collecting nearly 4,000 annotations on a domain-specific corpus of philosophy of science publications. Our findings reveal that word intrusion and topic coherence metrics do not always align, particularly in specialized domains, and that TWM captures human-perceived topic distinctness and aligns with diversity. 

## Key Contributions

- **Novel Evaluation Task**: Introduction of Topic Word Mixing (TWM) to assess inter-topic distinctness through human judgment
- **Comprehensive Evaluation**: Comparison of 6 topic models using both automatic metrics and human evaluations
- **Domain-Specific Dataset**: Nearly 4,000 annotations on philosophy of science publications
- **Open Resources**: Annotated dataset and task generation code for reproducible research

## Repository Structure

```
.
├── data/
│   ├── annotations/          # Human annotations collected
│   │   ├── word_intrusion_annotations.csv
│   │   └── word_mixing_annotations.csv
│   └── tasks/                # Generated evaluation tasks
│       ├── word_intrusion/   # Word intrusion tasks (4 tracks)
│       └── word_mixing/      # Word mixing tasks (4 tracks)
│
├── word_intrusion_and_mixing/
│   ├── streamlit_app.py      # Interactive annotation interface
│   ├── word_intrusion/       # Word intrusion implementation
│   │   ├── cli.py
│   │   ├── core.py
│   │   ├── processors.py
│   │   └── word_check.py
│   └── topic_mixing/         # Topic word mixing implementation
│       ├── core.py
│       └── processors.py
│
└── README.md                 # This file
```

## Dataset

The dataset contains human annotations for two complementary evaluation tasks:

### Word Intrusion Task
Traditional task where annotators identify the "intruder" word that doesn't belong with others from the same topic. This evaluates **intra-topic coherence**.

- **File**: `data/annotations/word_intrusion_annotations.csv`
- **Tasks**: `data/tasks/word_intrusion/` (4 tracks)

### Topic Word Mixing Task (TWM)
Novel task where annotators determine whether a set of words comes from a single topic or multiple mixed topics. This evaluates **inter-topic distinctness**.

- **File**: `data/annotations/word_mixing_annotations.csv`
- **Tasks**: `data/tasks/word_mixing/` (4 tracks)

### Annotation Statistics
- **Total annotations**: ~4,000
- **Domain**: Philosophy of science publications
- **Models evaluated**: LDA, NMF, Top2Vec, BERTopic, CFMF, CFMF-emb
- **Tracks**: 4 difficulty/variation tracks per task

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd LREC_2026_Prouteau_Human_Metrics_Topic_Models

# Install dependencies
cd word_intrusion_and_mixing
pip install -r requirements.txt
```

## Usage

### Running the Annotation Interface

Launch the interactive Streamlit application for performing annotations:

```bash
cd word_intrusion_and_mixing
./run_app.sh
```

Or manually:

```bash
streamlit run streamlit_app.py
```

The interface allows you to:
- Perform word intrusion tasks
- Perform topic word mixing tasks
- Switch between different tracks
- Save annotations automatically

### Task Generation

The repository includes tools for generating evaluation tasks from topic model outputs:

```python
from word_intrusion.core import generate_word_intrusion_tasks
from topic_mixing.core import generate_word_mixing_tasks

# Generate word intrusion tasks
intrusion_tasks = generate_word_intrusion_tasks(topics, num_tasks=50)

# Generate word mixing tasks
mixing_tasks = generate_word_mixing_tasks(topics, num_tasks=50)
```

### Data Format

#### Word Intrusion Tasks
Each task contains a set of words where one is an intruder:
```csv
task_id,model,topic_id,words,intruder
1,LDA,5,"word1,word2,word3,word4,intruder_word",intruder_word
```

#### Word Mixing Tasks
Each task asks if words are from one topic or mixed:
```csv
task_id,model,words,is_mixed,source_topics
1,BERTopic,"word1,word2,word3,word4",True,"topic3,topic7"
```

## Key Findings

Our evaluation reveals several important insights:

1. **Metric Misalignment**: Word intrusion results and topic coherence metrics do not always align, especially in specialized domains
2. **TWM Effectiveness**: Topic Word Mixing captures human-perceived topic distinctness and aligns well with diversity metrics
3. **Domain Specificity**: Specialized domains require more nuanced evaluation beyond standard automatic metrics
4. **Complementary Tasks**: Word intrusion (intra-topic) and TWM (inter-topic) provide complementary perspectives on topic quality

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{anonymous2026numbers,
  title={When Numbers Tell Half the Story: Human-Metric Alignment in Topic Model Evaluation},
  author={Anonymous},
  booktitle={Proceedings of LREC 2026},
  year={2026}
}
```

## Additional Documentation

- **[MANIFEST.md](word_intrusion_and_mixing/MANIFEST.md)**: Detailed file manifest
- **[PACKAGE_README.md](word_intrusion_and_mixing/PACKAGE_README.md)**: Package-specific documentation
- **[STOPWORD_ANALYSIS_README.md](word_intrusion_and_mixing/STOPWORD_ANALYSIS_README.md)**: Stopword analysis tools
- **[STOPWORD_TOOLS_GUIDE.md](word_intrusion_and_mixing/STOPWORD_TOOLS_GUIDE.md)**: Guide for stopword processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Add contact information here]

## Acknowledgments

We thank all annotators who contributed to this evaluation dataset and the open-source community for the topic modeling tools evaluated in this work.

---

**Note**: This is an anonymous submission for LREC 2026. Author information will be added upon acceptance.
