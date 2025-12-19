import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# --------------------------------------------------------------------
# 1. SETUP PATHS
# --------------------------------------------------------------------
# This is the path to the folder where you pasted your files
adapter_path = "./model" 

# This is the base model your adapters were trained on
base_model_name = "mistralai/Mistral-7B-v0.1"

print("⏳ Loading model... this might take a minute...")

# --------------------------------------------------------------------
# 2. LOAD BASE MODEL
# --------------------------------------------------------------------
# We load the base Mistral model first
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    low_cpu_mem_usage=True
    # load_in_4bit=True  # Uncomment this line if you have a GPU and installed bitsandbytes
)

# --------------------------------------------------------------------
# 3. LOAD YOUR FINE-TUNED ADAPTERS
# --------------------------------------------------------------------
# Now we merge your fine-tuned "brain" into the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() # Optimizes the model for speed

# Load the tokenizer (converts text to numbers)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully!")

# --------------------------------------------------------------------
# 4. DEFINE THE CHAT FUNCTION
# --------------------------------------------------------------------
def ask_medical_bot(message, history):
    # Format the prompt exactly like we did in training: <s>[INST] ... [/INST]
    prompt = f"<s>[INST] {message} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,    # Adjust length of answer
        temperature=0.3,       # Lower = more factual, Higher = more creative
        do_sample=True,
        top_p=0.9
    )
    
    # Decode the result (convert numbers back to text)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response to remove the original question
    # The model often repeats the prompt, so we split by [/INST] and take the second part
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()
        
    return response

# --------------------------------------------------------------------
# 5. CREATE THE UI (GRADIO)
# --------------------------------------------------------------------
# We use ChatInterface for a nice "ChatGPT-style" look
demo = gr.ChatInterface(
    fn=ask_medical_bot,
    title="🏥 Medical Assistant AI",
    description="Ask me about symptoms, treatments, or medical advice. (Powered by Fine-Tuned Mistral 7B)",
    examples=["What is the treatment for acute bronchitis?", "What are the symptoms of flu?", "How do I treat a burn?"],
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    demo.launch()