# PointGANet: A Lightweight 3D Point Cloud Learning Architecture for Semantic Segmentation

Welcome to the official repository for **PointGANet**, a pioneering approach in 3D point cloud semantic segmentation. This repository contains the implementation of PointGANet, an efficient and lightweight model designed to overcome the complexities associated with traditional point cloud processing architectures. 

## Overview

PointGANet introduces a novel grouped attention mechanism within an encoder featuring grouped convolution combined with element-wise multiplication. This significantly enhances feature extraction and emphasizes relevant features. Additionally, the decoder utilizes mini pointnet modules instead of the standard unit pointnet modules, resulting in a substantial reduction in trainable parameters without compromising accuracy. 

## Key Features

- **Lightweight Architecture**: PointGANet is approximately five times lighter than PointNet++ while maintaining high accuracy.
- **Grouped Attention Mechanism**: Enhances feature extraction capabilities.
- **Optimized Decoder**: Reduces trainable parameters using mini pointnet modules.
- **High Performance**: Achieves noteworthy improvements in mean accuracy and mean Intersection over Union (IoU) on the DALES dataset.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- MATLAB R2020b

### Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/mthanhlm/pointganet.git
cd pointganet
```

### Dataset

The DALES dataset is used for training and evaluation. You can download the dataset from [here](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php).


## Research Paper

This repository is based on the research paper titled [**PointGANet: A Lightweight 3D Point Cloud Learning Architecture for Semantic Segmentation**](https://doi.org/10.1145/3628797.3628809).

**Abstract**:
> PointNet++ has gained significant acknowledgement for point cloud data processing capabilities. Over time, various network improvements have been developed to enhance its global learning efficiency, thus boosting the correct segmentation rate. However, these improvements have often resulted in a significant increase in complexity, i.e., the model size and the processing speed. Meanwhile, improvements that focus on complexity reduction while preserving accuracy have been relatively scarce, particularly compared to some simpler models like SqueezeSegV2. To overcome this challenge, we embark on the development of a compact version of the PointNet++ model, namely PointGANet, tailored specifically for three-dimensional point cloud semantic segmentation. In PointGANet, we introduce a grouped attention mechanism in an encoder with grouped convolution incorporated with element-wise multiplication to enrich feature extraction capability and emphasise relevant features. In a decoder, we replace unit pointnet modules with mini pointnet modules to save a massive number of trainable parameters. Through rigorous experimentation, we successfully fine-tune the network to obtain a significant reduction in model size while maintaining accuracy, hence resulting in a substantial enhancement in overall performance. Remarkably, relying on the intensive evaluation using the DALES dataset, PointGANet is more lightweight than the original PointNet++ by approximately five times with some noteworthy improvements in mean accuracy by and mean IoU. These innovations open up exciting possibilities for developing point cloud segmentation applications on IoT and resource-constrained devices.

## Contributions

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to all the contributors and the community for their valuable feedback and suggestions.

---

We hope PointGANet serves as a valuable resource for your 3D point cloud semantic segmentation projects. Happy coding!
