# CBMAFF-Net
CBMAFF-Net is an advanced deep learning model designed for the non-targeted screening of new psychoactive substances (NPS) using Nuclear Magnetic Resonance (NMR) data. The model combines Convolutional Neural Networks (CNN) to extract spatial features and Bidirectional Long Short-Term Memory (BiLSTM) networks to capture temporal dependencies within the NMR spectra. Additionally, an attention mechanism is incorporated to focus on the most relevant parts of the data, enhancing the model's accuracy and interpretability.

By integrating these components, CBMAFF-Net effectively captures both local and global information from NMR data, leading to superior performance in identifying NPS, even when dealing with novel or undocumented substances. The model has demonstrated consistent outperformance of current state-of-the-art methods across multiple independent datasets, making it a valuable tool for forensic science and regulatory applications.
# Usage
1.The data used in this study came from the collection of the original database, to which auxiliary materials were first added to generate the simulated dataset, and the code was run as shown below:
~~~
python data_processing/gen_mixture.py --input_file_path data/fuliao/fuliao.xlsx --sheet_name_C C --sheet_name_H H --excluded_numbers 5 10 --iterations 25 --fraction 0.2 --output_file_path data/fuliao/processed_data.xlsx
~~~
2.To convert a simulated dataset into vectorized data that can be input into a model, the following code can be used:
~~~
python data_processing/feature_extraction.py
~~~
