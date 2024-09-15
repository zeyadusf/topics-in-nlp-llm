<div align= "center">  

  # PEFT (Parameter-Efficient Fine-Tuning)


<img src="https://github.com/user-attachments/assets/9af04066-d2af-401d-ad35-00b41954de74" width="800" height="600">
</div>

## :books: Table of Contents
<ol>
  
  <li><a href="#about">What is PEFT ?</a></li>
  <li><a href="#advantages">Why is it important?</a></li>
  <li><a href="#lib">How does PEFT work?</a></li>
  <li><a href="#types">PEFT Methods</a></li>
  <li><a href="q">Questions</a></li>
  <li><a href="#projects">Projects</a></li>
  <li><a href="#resourses">Resources</a></li>

</ol>
<br>
<br>
<div id="about">

# What is PEFT ?

Parameter-Efficient Fine-Tuning (PEFT) is a technique designed to fine-tune large pre-trained language models with minimal computational and memory resources. Instead of updating all the model's parameters, PEFT selectively adjusts a small subset of parameters, significantly reducing storage and computational costs while achieving performance levels comparable to fully fine-tuned models. This approach is particularly useful for efficiently adapting large language models (LLMs) to downstream tasks without the prohibitive costs typically associated with fine-tuning. PEFT integrates seamlessly with frameworks like Transformers, Diffusers, and Accelerate, enabling easy model training, distributed inference, and managing different adapters for large-scale models.
</div>
<br>
<div id="advantages">

# Why is it important?
There are many benefits of using PEFT but the main one is the huge savings in compute and storage, making PEFT applicable to many different use cases.
- **Reduced Training Costs:** By updating fewer parameters, PEFT significantly reduces the computational and memory requirements for fine-tuning large models, making it possible to fine-tune models even on limited hardware like GPUs with smaller memory.
- **Faster Training:** Since fewer parameters are being adjusted, training can be faster compared to full fine-tuning.
- **Memory Efficiency:** PEFT uses less memory during the fine-tuning process as the base model's parameters remain frozen, and only a small portion of the parameters (or additional components) are trained.
- **Modular and Adaptable:** PEFT methods allow fine-tuning across multiple tasks with minimal adjustments, often enabling model sharing without requiring the entire model to be fine-tuned again.
</div>
<br>
<div id="lib">
  
# How does PEFT work?
- **Freezing Pre-trained Model Weights:** The key idea is that the core model weights remain frozen and do not require updates. This saves significant resources.
- **Introducing Small, Trainable Components:** PEFT methods typically introduce small trainable components (like adapters or low-rank matrices) that are easy to adjust without modifying the entire model.
- **Fine-tuning the Subset:** During training, only these additional components (or certain parameter subsets, like biases) are updated while the rest of the model stays unchanged. This allows for the model to learn task-specific information efficiently.

## Quickstart: 
PEFT works by using a library that enables the selective fine-tuning of LLMs. Instead of tuning the entire model, only a subset of parameters is adjusted, reducing the computational burden. This is achieved through techniques like Low-rank Adaptation (LoRA), which decomposes the model's weight matrices to reduce the number of trainable parameters.

> Install PEFT from pip :
```py
pip install peft
```
> Prepare a model for training with a PEFT method such as _**LoRA**_ by wrapping the base model and PEFT configuration with get_peft_model. For the bigscience/mt0-large model, you're only training 0.19% of the parameters!
```py
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"
```
> To load a PEFT model for inference:
```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
```
</div>
<br>
<div id="types">
  
# PEFT Methods

<img src="https://github.com/user-attachments/assets/1fe1b5a5-db80-4428-bd91-3d190b0e7c93"  height="720">

The figure represents the evolutionary development of **Parameter-Efficient Fine-Tuning (PEFT)** methods . It categorizes various PEFT approaches into distinct branches based on their fine-tuning mechanisms. Here’s an explanation of the colored branches and notable techniques in each:

### 1. **Additive Fine-Tuning (Green Branch)**
   - **Characteristics**: In this approach, additional parameters (such as adapters or prompts) are added to the model, and only these new components are fine-tuned while the original model remains frozen.
   - **Notable Techniques**:
     - **Prefix-tuning** (2021): Adds trainable prefix tokens to the input, allowing task adaptation without altering the model weights.
     - **Prompt-tuning** (2021): Focuses on learning task-specific prompts that are prepended to inputs to steer the model's performance.
     - **HyperFormer++** (2022): A more advanced variant of HyperFormer, used in efficient transfer learning for multiple tasks.

### 2. **Reparameterized Fine-Tuning (Orange Branch)**
   - **Characteristics**: Instead of adding new parameters, this method involves reparameterizing certain components of the model (e.g., weight matrices) to introduce a low-rank approximation, optimizing the number of trainable parameters.
   - **Notable Techniques**:
     - **LoRA (Low-Rank Adaptation)** (2021): Introduces low-rank matrices into the transformer layers, enabling efficient fine-tuning with fewer parameters. LoRA has become one of the most widely used techniques for fine-tuning large language models.
     - **Delta-LoRA** (2023): A more advanced version of LoRA that further optimizes the efficiency of fine-tuning by modifying the delta updates.

### 3. **Unified Fine-Tuning (Purple Branch)**
   - **Characteristics**: A unified approach where multiple fine-tuning methods are combined or adapted into a single, flexible framework.
   - **Notable Techniques**:
     - **ProPETL** (2023): This technique combines various PEFT methods into a unified framework, making it easier to experiment and adapt models across tasks.
     - **Sparse Adapter** (2023): A method that introduces sparsity constraints to fine-tuning, reducing the number of parameters being adjusted.

### 4. **Hybrid Fine-Tuning (Red Branch)**
   - **Characteristics**: Combines techniques from different fine-tuning strategies (such as additive and reparameterized methods) to leverage the strengths of both approaches.
   - **Notable Techniques**:
     - **S⁴ (2023)**: A hybrid fine-tuning method that mixes different strategies for more efficient learning across various tasks.

### 5. **Partial Fine-Tuning (Blue Branch)**
   - **Characteristics**: In partial fine-tuning, only specific parts of the model, like the bias terms or certain layers, are fine-tuned, rather than adjusting all model parameters.
   - **Notable Techniques**:
     - **BitFit** (2021): Only the bias terms of the model are fine-tuned, making it a lightweight method for task adaptation.
     - **UniPELT** (2023): Combines partial fine-tuning with other PEFT methods for a more comprehensive fine-tuning approach.


> Summary of Most Famous Techniques per Branch:
> - **Additive Fine-Tuning**: **Prefix-tuning** and **Prompt-tuning** are widely used due to their simplicity and flexibility in adapting models to various tasks.
> - **Reparameterized Fine-Tuning**: **LoRA** is the standout technique for efficiently fine-tuning large models while maintaining performance.
> - **Unified Fine-Tuning**: **ProPETL** integrates multiple fine-tuning methods into a single adaptable framework.
> - **Hybrid Fine-Tuning**: **S⁴** merges methods for a more flexible and efficient fine-tuning approach.
> - **Partial Fine-Tuning**: **BitFit** is a well-known, lightweight method that fine-tunes only the bias terms of a model.
Each branch reflects different strategies to reduce the computational cost of fine-tuning large models while maintaining task performance.
</div>
<br>
<div id="q">

# Questions? 

<h3 style="font-family:'Georgia',serif; font-size: 1.2em;color:#982B1C;margin-bottom: 10px;">❓ Is "Quantization" a method of "PEFT" ?</h3>

>  **Quantization** is not a **PEFT (Parameter-Efficient Fine-Tuning)** method but can be used alongside it to improve efficiency.

> **Quantization** is the process of reducing the precision of numerical values (such as weights and activations) in an AI model. The goal is to reduce the size of the model and speed up computations, making models more memory and power efficient.

> ### How is Quantization different from PEFT?
> - **PEFT** focuses on reducing the number of parameters that are updated during the training process, such as LoRa technology that updates a small portion of the weight matrices.
> - **Quantization** works to reduce the precision of the weights or activations in the model to reduce memory consumption and speed up computations, such as  reducing the weights from 32 bits to 8 bits or 4 bits.

> ### Relationship between PEFT and Quantization:
> - **QLoRa** is an example of combining the two methods. QLoRa uses Quantization to reduce the precision of the weights to 4 bits, and then applies LoRa to update a small portion of the weights via low-order matrices, providing maximum efficiency in memory and computation. In short, **Quantization** is not a ?  direct part of PEFT but can be used alongside it to achieve higher efficiency in training and optimization, as in QLoRa.

<b> :link: [What is Quantization?]()</b>

---
<h3 style="font-family:'Georgia',serif; font-size: 1.2em;color:#982B1C;margin-bottom: 10px;">❓ What is LoRA?</h3>
<br>

<b> :link: [Click here to know about LoRA]()</b>
<hr>
<br>
<hr>
<hr>

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
    <th>Method</th>
  </tr>

  <tr>
    <td>1</td>
    <td>Summarization-by-Finetuning-FlanT5-LoRA</td> 
    <td><b>FlanT5</b></td>
    <td><b>Summarization</b></td>
    <td><a href="https://github.com/zeyadusf/Summarization-by-Finetuning-FlanT5-LoRA">Summarization-by-Finetuning-FlanT5-LoRA</a></td>
    <td><a href="https://www.kaggle.com/code/zeyadusf/summarization-by-finetuning-flant5-lora">Summarization by Finetuning FlanT5-LoRA</a></td>
    <td><a href="https://huggingface.co/zeyadusf/FlanT5Summarization-samsum">FlanT5Summarization-samsum </a></td>
    <td><a href="https://huggingface.co/spaces/zeyadusf/Summarizationflant5">Summarization by Flan-T5-Large with PEFT</a></td>
    <td>
      <i>use "LoRA"</i><br>
    </td>
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
      <td>
      <i>use "LoRA"</i><br>
    </td>
  </tr>
  </table>
</div>

<br>
<div id="resourses">

# Resources:

> - [Github : PEFT ](https://github.com/huggingface/peft)
> - [Medium : Fine-Tune LLM with PEFT](https://medium.com/@MUmarAmanat/fine-tune-llm-with-peft-60b2798f1e5f)
> - [Arxiv : Parameter-Efficient Fine-Tuning Methods forPretrained Language Models](https://arxiv.org/pdf/2312.12148)
> - [Snorkel : LoRA: Low-Rank Adaptation for LLMs](https://snorkel.ai/lora-low-rank-adaptation-for-llms/#h.mbbmv22kacdj)
