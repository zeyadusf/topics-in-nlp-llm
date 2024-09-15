<div align='center'>

  # Low Rank Adaptation

Parameter-efficient LLM fine-tuning with LoRA
![lora](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GmRC9gHJy1maqcFP7IRDcg.png)
</div>

## :books: Table of Contents
<ol>
  
  <li><a href="#about">What is LoRA ?</a></li>
  <li><a href="#advantages">Why is it important?</a></li>
  <li><a href="#lib">How does PEFT work?</a></li>
  <li><a href="#code">How to use LoRA</a></li>
  <li><a href="#projects">Projects</a></li>
</ol>

<div id='about'>
  
# What is LoRA :
  LoRa is a technique for efficiently fine-tuning large language models (LLMs).
  The key idea is to freeze the pretrained model's weights and only update a low-rank decomposition of some weight matrices. 
  This reduces the number of trainable parameters, making fine-tuning more memory-efficient.

> **This type of Reparameterized Fine-Tuning (Orange Branch) that was mentioned in [`PEFT`](https://github.com/zeyadusf/topics-in-nlp-llm/edit/main/PEFT%20(Parameter-Efficient%20Fine-Tuning)/README.md)**

</div>
<br>
<br>
<div id='advantages'>

# Advantages of Using LoRA for Fine-Tuning :

**Efficiency in Training and Adaptation**

>LoRA enhances the training and adaptation efficiency of large language models like OpenAI’s GPT-3 and Meta’s LLaMA. Traditional fine-tuning methods require updating all model parameters, which is computationally intensive. LoRA, instead, introduces low-rank matrices that only modify a subset of the original model's weights. These matrices are small compared to the full set of parameters, enabling more efficient updates.
>
>The approach focuses on altering the weight matrices in the transformer layers of the model, specifically targeting the most impactful parameters. This selective updating streamlines the adaptation process, making it significantly quicker and more efficient. It allows the model to adapt to new tasks or datasets without the need to extensively retrain the entire model.

**Reduced Computational Resources Requirement**
>LoRA reduces the computational resources required for fine-tuning large language models. By using low-rank matrices to update specific parameters, the approach drastically cuts down the number of parameters that need to be trained. This reduction is crucial for practical applications, as fully retraining LLM models like GPT-3 is beyond the resource capabilities of most organizations.
>
>LoRA's method requires less memory and processing power, and also allows for quicker iterations and experiments, as each training cycle consumes fewer resources. This efficiency is particularly beneficial for applications that require regular updates or adaptations, such as adapting a model to specialized domains or continuously evolving datasets.

**Preservation of Pre-trained Model Weights**
>LoRA preserves the integrity of pre-trained model weights, which is a significant advantage. In traditional fine-tuning, all weights of the model are subject to change, which can lead to a loss of the general knowledge the model originally possessed. LoRA's approach of selectively updating weights through low-rank matrices ensures that the core structure and knowledge embedded in the pre-trained model are largely maintained.
>
>This preservation is crucial for maintaining the model's broad understanding and capabilities while still allowing it to adapt to specific tasks or datasets. It ensures that the fine-tuned model retains the strengths of the original model, such as its understanding of language and context, while gaining new capabilities or improved performance in targeted areas
</div>
<br>
<br>
<div id="lib">

# How LoRA Works ?

![image](https://github.com/user-attachments/assets/ad74500c-7e1e-4cc7-9178-f66888c4c2aa)

LoRA approximates the weight updates through two much smaller matrices, 
𝐴 and 𝐵, where the rank 𝑟 of the decomposition is significantly smaller than the dimensions of the original weight matrix. The approach can be described as follows: the pre-trained weights 𝑊₀ are kept frozen, and only the low-rank matrices are trained, with the weight updates Δ𝑊 being approximated as 𝐵⋅𝐴. The final weight matrix after fine-tuning is calculated as:

𝑊 = 𝑊₀ + 𝐵⋅𝐴

This maintains the same dimensionality as the original weights while reducing the number of trainable parameters. This method leads to substantial savings in memory and computational resources, especially when fine-tuning large models like GPT-3, allowing for faster training with fewer GPUs while retaining model performance.

### Summary of Algorithm:
1. **Input**: Pre-trained weight matrix **W₀**, rank **r**, input data **x**
2. **Step 1**: Decompose **ΔW** into **B ⋅ A** with low-rank matrices **B ∈ Rᵈˣʳ** and **A ∈ Rʳˣᵏ**
3. **Step 2**: Fine-tune **B** and **A** while keeping **W₀** frozen
4. **Step 3**: Compute the final weight matrix **W = W₀ + B ⋅ A**
5. **Output**: Fine-tuned weight matrix **W**

</div>
<br>
<br>
<div id='code'>

# How to Use LoRA?

### 1. Install the PEFT library
First, you need to install the PEFT library which supports LoRA. Use pip to install it:
```py
pip install peft
```
### 2. Import Dependencies
To use LoRA, you'll need to import the required classes and methods from the peft library:
```py
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSeq2SeqLM
```

### 3. Set LoRA Configuration
Define the configuration for LoRA. Here’s an example for a sequence-to-sequence language model (Seq2SeqLM):
```py
# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # Target specific attention modules
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
```
### 4. Prepare the Base Model with LoRA
Once you have defined the configuration, apply LoRA to the base model:
```py
# Load a pre-trained model (for example, T5)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")

# Add the LoRA adapter
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters to confirm LoRA was applied
model.print_trainable_parameters()
# Example Output: trainable params: 4,718,592 || all params: 787,868,672 || trainable%: 0.5989
```
</div>
<br>


<div id="projects">

# Projects

<table style="width:100%">
  <tr>
    <th>NO.</th>
    <th>Project Name</th>
    <th>Model Name</th>
    <th>Task</th>
    <th>GitHub</th>
    <th>Kaggle</th>
    <th>Hugging Face</th>
    <th>Space</th>


  <tr>
    <td>1</td>
    <td>Summarization-by-Finetuning-FlanT5-LoRA</td> 
    <td><b>FlanT5</b></td>
    <td><b>Summarization</b></td>
    <td><a href="https://github.com/zeyadusf/Summarization-by-Finetuning-FlanT5-LoRA">Summarization-by-Finetuning-FlanT5-LoRA</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/summarization-by-finetuning-flant5-lora">Summarization by Finetuning FlanT5-LoRA</a></td>
    <td><a href="https://huggingface.co/zeyadusf/FlanT5Summarization-samsum">FlanT5Summarization-samsum </a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Summarizationflant5">Summarization by Flan-T5-Large with PEFT</a></td>
  </tr>

  <tr>
    <td>2</td>
    <td>Finetune Llama2</td> 
    <td><b>Llama2</b></td>
    <td><b>Text Generation</b></td>
    <td><a href="https://github.com/zeyadusf/FineTune-Llama2">FineTune-Llama2</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/finetune-llama2">FineTune-Llama2</a></td>
    <td><a href="https://huggingface.co/zeyadusf/llama2-miniguanaco">llama2-miniguanaco </a></td>
    <td><a href=#">---</a></td>
  </tr>
  </table>
</div>

<br>
