# Quantization in Depth

[![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-Course-blue)](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Compatible-yellow)](https://huggingface.co/)

This repository contains materials and code examples from the DeepLearning.AI short course "Quantization in Depth", instructed by Marc Sun and Younes Belkada from Hugging Face.

## 📚 Course Overview

"Quantization in Depth" teaches advanced techniques to compress neural network models, reducing their size to a fraction of the original while maintaining performance. By implementing customized quantization methods from scratch, you'll gain deep insights into the tradeoffs between model size and accuracy, enabling faster inference and broader deployment of AI models.

## 🎯 What You'll Learn

- Implement and compare different variants of linear quantization (symmetric vs. asymmetric mode)
- Apply varying granularity levels: per-tensor, per-channel, and per-group quantization
- Build a general-purpose quantizer in PyTorch that can compress any open source model's dense layers by up to 4x
- Implement weights packing techniques to compress weights from 32 bits to as low as 2 bits

## 🗂️ Course Structure

This course includes 18 lessons with 13 code examples:

### Fundamentals of Quantization
1. **Introduction** - Overview of quantization importance and applications
2. **Overview** - Core concepts and techniques covered in the course

### Building Quantization from Scratch
3. **Quantize and De-quantize a Tensor** - Fundamental operations in quantization
4. **Get the Scale and Zero Point** - Understanding key quantization parameters
5. **Symmetric vs Asymmetric Mode** - Comparing different quantization approaches
6. **Finer Granularity for more Precision** - Introduction to granularity concepts

### Advanced Granularity Techniques
7. **Per Channel Quantization** - Implementing channel-wise quantization
8. **Per Group Quantization** - Implementing group-wise quantization
9. **Quantizing Weights & Activations for Inference** - Practical application to inference

### Building a Complete Quantizer
10. **Custom Build an 8-Bit Quantizer** - Developing a custom quantization solution
11. **Replace PyTorch layers with Quantized Layers** - Practical integration with PyTorch
12. **Quantize any Open Source PyTorch Model** - Building a general-purpose solution
13. **Load your Quantized Weights from HuggingFace Hub** - Working with the Hugging Face ecosystem

### Ultra-Low Bit Quantization
14. **Weights Packing** - Theory behind extreme compression
15. **Packing 2-bit Weights** - Implementation of 2-bit weight compression
16. **Unpacking 2-Bit Weights** - Recovering usable weights from compressed format

### Looking Forward
17. **Beyond Linear Quantization** - Introduction to advanced quantization methods
18. **Conclusion** - Summary and future directions

## 💻 Code Examples

This repository contains 13 code examples that correspond to the course lessons:

1. **Tensor Quantization** - Basic quantize/dequantize operations
2. **Computing Scale and Zero Point** - Determining quantization parameters
3. **Symmetric vs. Asymmetric Modes** - Implementing both quantization modes
4. **Per-Tensor Quantization** - Basic granularity implementation
5. **Per-Channel Quantization** - Channel-wise implementation
6. **Per-Group Quantization** - Group-wise implementation
7. **Weights & Activations Quantization** - Full inference-ready quantization
8. **8-Bit Quantizer Implementation** - Complete 8-bit solution
9. **Quantized Layer Replacement** - Integration with PyTorch
10. **General-Purpose Model Quantizer** - Quantizing any PyTorch model
11. **Hugging Face Integration** - Loading and saving quantized weights
12. **2-Bit Weight Packing** - Ultra-low bit compression implementation
13. **2-Bit Weight Unpacking** - Efficient decompression implementation

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/duybaohuynhtan/Quantization-in-Depth.git
cd Quantization-in-Depth

# Install required packages
pip install -r requirements.txt
```

### Running the Examples

Each code example is presented as a Jupyter notebook:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open the desired example.

## 📋 Requirements

- Python 3.9.18
- Accelerate==0.26.1
- Seaborn==0.13.1
- Torch==2.1.1
- Transformers==4.35.0

## 🔗 Additional Resources

- [DeepLearning.AI Short Course Link](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 👨‍🏫 Instructors

- [**Marc Sun**](https://www.linkedin.com/in/marc-sun/) - Machine Learning Engineer at Hugging Face
- [**Younes Belkada**](https://www.linkedin.com/in/younes-belkada-b1a903145/?locale=en_US) - Machine Learning Engineer at Hugging Face

## 🙏 Acknowledgments

Special thanks to DeepLearning.AI and Hugging Face for creating this educational content.