# SwinTrack Object Tracking Inference

This script provides a complete solution for object tracking using SwinTrack on image sequences. It reads images from a folder, implements SwinTrack for object tracking, and saves the processed outputs with track vectors in a separate folder.

## Features

- **Object Tracking**: Uses SwinTrack model for robust object tracking
- **Track Vector Visualization**: Shows movement vectors between frames
- **Response Map Analysis**: Saves and visualizes response maps
- **Batch Processing**: Processes entire image sequences automatically
- **Flexible Input**: Supports various image formats (JPG, PNG, BMP, TIFF)
- **Configurable**: Easy to modify tracking parameters

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_inference.txt
```

2. Make sure you have a pretrained SwinTrack model weights file (.pth)

## Usage

### Basic Usage

```bash
python swintrack_inference.py \
    --input_folder /path/to/input/images \
    --output_folder /path/to/output \
    --model_path /path/to/swintrack_model.pth
```

### Advanced Usage

```bash
python swintrack_inference.py \
    --input_folder /path/to/input/images \
    --output_folder /path/to/output \
    --model_path /path/to/swintrack_model.pth \
    --config_path config/SwinTrack/Base/config.yaml \
    --device cuda \
    --initial_bbox 0.3 0.3 0.2 0.2
```

### Parameters

- `--input_folder`: Path to folder containing input images (required)
- `--output_folder`: Path to folder for saving outputs (required)
- `--model_path`: Path to pretrained SwinTrack model weights (required)
- `--config_path`: Path to SwinTrack config file (default: config/SwinTrack/Base/config.yaml)
- `--device`: Device to run inference on - 'cuda' or 'cpu' (default: cuda)
- `--initial_bbox`: Initial bounding box in normalized coordinates (x, y, w, h) (optional)

## Output Structure

The script creates the following output structure:

```
output_folder/
├── visualizations/          # Visualization images with bounding boxes and track vectors
│   ├── vis_0000.jpg
│   ├── vis_0001.jpg
│   └── ...
├── response_maps/           # Response maps as numpy arrays
│   ├── response_0000.npy
│   ├── response_0001.npy
│   └── ...
└── tracking_results.json    # JSON file with tracking results and metadata
```

## Output Files

### Visualizations
- **vis_XXXX.jpg**: Images showing tracking results with:
  - Red bounding box around tracked object
  - Blue arrow showing movement vector
  - Response map heatmap

### Response Maps
- **response_XXXX.npy**: Numpy arrays containing the raw response maps from SwinTrack

### Tracking Results
- **tracking_results.json**: JSON file containing:
  - Frame numbers
  - Image paths
  - Bounding box coordinates (normalized)
  - Response map file paths

## Example

```python
# Example of using the SwinTrackInference class directly
from swintrack_inference import SwinTrackInference

# Initialize tracker
tracker = SwinTrackInference(
    config_path='config/SwinTrack/Base/config.yaml',
    model_path='swintrack_model.pth',
    device='cuda'
)

# Process a single image sequence
image_paths = ['frame1.jpg', 'frame2.jpg', 'frame3.jpg']
tracker.process_image_sequence(
    image_paths=image_paths,
    output_dir='output',
    initial_bbox=(0.3, 0.3, 0.2, 0.2)  # x, y, w, h in normalized coordinates
)
```

## Notes

1. **Model Weights**: You need to provide pretrained SwinTrack model weights. These can be obtained from the official SwinTrack repository.

2. **Initial Bounding Box**: If not provided, the script will use the center of the first image as the initial tracking location.

3. **Image Formats**: Supports common image formats including JPG, PNG, BMP, and TIFF.

4. **GPU Memory**: For large images or long sequences, ensure sufficient GPU memory is available.

5. **Config File**: The default config file is included in the repository. You can modify it to adjust tracking parameters.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce image size or use CPU inference
2. **Model Loading Error**: Ensure the model weights file is compatible with the config
3. **No Images Found**: Check that the input folder contains supported image formats

### Performance Tips

1. Use GPU inference for better performance
2. Resize images to smaller dimensions if memory is limited
3. Process images in batches for better efficiency

## License

This script is part of the SwinTrack project. Please refer to the original SwinTrack license for usage terms.